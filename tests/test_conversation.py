"""
Tests for conversation memory and follow-up query rewriting.

The rewriter is rule-based on purpose, so these tests pin exact
behaviour rather than asserting a model's phrasing.
"""
import pytest

from app.generation.context_builder import ContextBuilder
from app.generation.conversation import (
    MAX_HISTORY_CHARS,
    MAX_HISTORY_TURNS,
    ConversationMemory,
    ConversationTurn,
    needs_rewrite,
    rewrite_query,
)
from app.generation.providers.ollama_provider import OllamaMultimodalProvider
from tests.test_pipeline import (
    SAMPLE_RESULTS,
    FakeRetriever,
    FakeStreamingGenerator,
    build_pipeline,
)


def memory(*turns: tuple[str, str]) -> ConversationMemory:
    return ConversationMemory(
        [
            ConversationTurn(role=role, content=content)
            for role, content in turns
        ]
    )


# --------------------------------------------------------------
# Memory bounds
# --------------------------------------------------------------
def test_memory_is_falsey_without_history():
    assert not ConversationMemory()
    assert not ConversationMemory(None)


def test_memory_keeps_only_the_most_recent_turns():
    turns = [
        ConversationTurn(role="user", content=f"question {index}")
        for index in range(MAX_HISTORY_TURNS + 4)
    ]

    kept = ConversationMemory(turns)

    assert len(kept.recent()) == MAX_HISTORY_TURNS
    # The tail is what a follow-up refers to.
    assert kept.recent()[-1].content == (
        f"question {MAX_HISTORY_TURNS + 3}"
    )


def test_memory_truncates_an_oversized_history_from_the_end():
    turns = [
        ConversationTurn(role="user", content="x" * MAX_HISTORY_CHARS)
        for _ in range(4)
    ]

    rendered = ConversationMemory(turns).as_prompt_context()

    assert len(rendered) <= MAX_HISTORY_CHARS + 1
    assert rendered.startswith("…")


def test_memory_renders_nothing_for_empty_turns():
    assert (
        ConversationMemory(
            [ConversationTurn(role="user", content="   ")]
        ).as_prompt_context()
        is None
    )


def test_previous_user_query_skips_assistant_turns():
    # The referent is the earlier *question*. Folding a model-written
    # answer into the search would let the model bias its own
    # retrieval.
    memory_ = memory(
        ("user", "What is the OptiPlex 3020?"),
        ("assistant", "It is a compact desktop with an Intel i5."),
    )

    assert memory_.previous_user_query == "What is the OptiPlex 3020?"


# --------------------------------------------------------------
# Detecting a follow-up
# --------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "What about its warranty?",
        "And the price?",
        "Tell me more",
        "Anything else?",
        "Does it have a warranty?",
        "How much is it?",
        "Why?",
    ],
)
def test_follow_ups_are_detected(query):
    assert needs_rewrite(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What is the Dell OptiPlex 3020 warranty?",
        "Which pages mention wireless connectivity?",
        "Tell me about the Dell Precision workstation tower",
    ],
)
def test_self_contained_queries_are_left_alone(query):
    # Rewriting a query that already names its subject would only
    # dilute it with unrelated earlier terms.
    assert needs_rewrite(query) is False


# --------------------------------------------------------------
# Rewriting
# --------------------------------------------------------------
def test_rewrite_prepends_the_previous_question():
    result = rewrite_query(
        "What about its warranty?",
        memory(("user", "What is the Dell OptiPlex 3020?")),
    )

    assert result.was_rewritten is True
    assert result.query == "What about its warranty?"
    # The model answers what the user actually typed...
    assert "OptiPlex" not in result.query
    # ...but retrieval searches for something standalone.
    assert "OptiPlex 3020" in result.search_query
    assert result.search_query.endswith("What about its warranty?")


def test_rewrite_is_a_no_op_without_history():
    result = rewrite_query("What about its warranty?", None)

    assert result.was_rewritten is False
    assert result.search_query == "What about its warranty?"


def test_rewrite_is_a_no_op_for_a_self_contained_query():
    original = "Which pages mention wireless connectivity?"

    result = rewrite_query(
        original,
        memory(("user", "What is the Dell OptiPlex 3020?")),
    )

    assert result.was_rewritten is False
    assert result.search_query == original


def test_rewrite_is_a_no_op_when_only_assistant_turns_exist():
    result = rewrite_query(
        "And the price?",
        memory(("assistant", "It has an Intel i5 processor.")),
    )

    assert result.was_rewritten is False


def test_rewrite_uses_the_most_recent_question():
    result = rewrite_query(
        "And its warranty?",
        memory(
            ("user", "What is the Dell OptiPlex 3020?"),
            ("assistant", "A compact desktop."),
            ("user", "What about the monitor?"),
        ),
    )

    assert "monitor" in result.search_query


# --------------------------------------------------------------
# Pipeline wiring
# --------------------------------------------------------------
def test_pipeline_searches_with_the_rewritten_query():
    retriever = FakeRetriever(results=SAMPLE_RESULTS)
    pipeline = build_pipeline(retriever, FakeStreamingGenerator())

    pipeline.run(
        "What about its warranty?",
        history=[
            ConversationTurn(
                role="user",
                content="What is the Dell OptiPlex 3020?",
            )
        ],
    )

    assert retriever.queries == [
        "What is the Dell OptiPlex 3020? What about its warranty?"
    ]


def test_pipeline_gives_the_model_the_original_wording():
    """
    The model must answer the question that was asked.

    Feeding it the rewritten search string would make it answer
    "what is the OptiPlex 3020 what about its warranty?" instead.
    """
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeStreamingGenerator(),
    )

    pipeline.run(
        "What about its warranty?",
        history=[
            ConversationTurn(
                role="user",
                content="What is the Dell OptiPlex 3020?",
            )
        ],
    )

    context = pipeline.generator.contexts[0]
    assert context.query == "What about its warranty?"
    assert "OptiPlex 3020?" in context.conversation


def test_pipeline_passes_no_conversation_for_a_single_turn():
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeStreamingGenerator(),
    )

    pipeline.run("What is the Dell OptiPlex 3020?")

    assert pipeline.generator.contexts[0].conversation is None


def test_streaming_path_rewrites_identically():
    retriever = FakeRetriever(results=SAMPLE_RESULTS)
    pipeline = build_pipeline(retriever, FakeStreamingGenerator())

    list(
        pipeline.run_stream(
            "What about its warranty?",
            history=[
                ConversationTurn(
                    role="user",
                    content="What is the Dell OptiPlex 3020?",
                )
            ],
        )
    )

    assert retriever.queries == [
        "What is the Dell OptiPlex 3020? What about its warranty?"
    ]


# --------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------
def test_context_builder_omits_an_empty_conversation():
    context = ContextBuilder().build(
        query="What is the Dell OptiPlex 3020?",
        results=SAMPLE_RESULTS,
    )

    assert context.conversation is None


def test_ollama_prompt_includes_the_conversation():
    provider = OllamaMultimodalProvider()
    captured: dict = {}

    provider.client.chat = lambda **kw: captured.update(kw) or {
        "message": {"content": "ok"}
    }

    provider.generate(
        query="What about its warranty?",
        text_context=[{"page": 1, "content": "text"}],
        image_context=[],
        conversation="User: What is the OptiPlex 3020?",
    )

    prompt = captured["messages"][1]["content"]
    assert "Earlier in this conversation" in prompt
    assert "What is the OptiPlex 3020?" in prompt
    # The current question is still the question.
    assert "What about its warranty?" in prompt


def test_ollama_prompt_omits_the_section_without_history():
    provider = OllamaMultimodalProvider()
    captured: dict = {}

    provider.client.chat = lambda **kw: captured.update(kw) or {
        "message": {"content": "ok"}
    }

    provider.generate(
        query="What is the OptiPlex 3020?",
        text_context=[{"page": 1, "content": "text"}],
        image_context=[],
    )

    assert "Earlier in this conversation" not in captured["messages"][1][
        "content"
    ]


def test_conversation_does_not_weaken_the_grounding_rules():
    """
    Earlier turns must never become admissible evidence.

    A model that treats a previous answer as a source can cite
    something the documents never said.
    """
    provider = OllamaMultimodalProvider()
    captured: dict = {}

    provider.client.chat = lambda **kw: captured.update(kw) or {
        "message": {"content": "ok"}
    }

    provider.generate(
        query="What about its warranty?",
        text_context=[{"page": 1, "content": "text"}],
        image_context=[],
        conversation=(
            "User: What is the OptiPlex 3020?\n"
            "Assistant: It has a 5-year warranty."
        ),
    )

    system_prompt = captured["messages"][0]["content"]
    assert "I don't have enough information" in system_prompt
    assert "for resolving references only" in captured["messages"][1][
        "content"
    ]

# --------------------------------------------------------------
# Pronoun detection
# --------------------------------------------------------------
def test_a_pronoun_marks_a_query_as_a_follow_up_even_when_it_is_long():
    """
    "Does it support Wi-Fi 6?" tokenizes to three words, which would
    otherwise look self-contained.

    Two things hide the missing subject: "it" is a stopword the index
    tokenizer drops, and "Wi-Fi" splits into two fragments that inflate
    the count. The pronoun check runs on raw words for that reason.
    """
    from app.retrieval.lexical import tokenize

    assert len(tokenize("Does it support Wi-Fi 6?")) >= 3
    assert needs_rewrite("Does it support Wi-Fi 6?") is True


@pytest.mark.parametrize(
    "query",
    [
        "Does it come with a monitor?",
        "How much memory does that have?",
        "Are they still available?",
        "Which of these support wireless?",
        "Is this the same model?",
    ],
)
def test_pronouns_trigger_a_rewrite(query):
    assert needs_rewrite(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What is the Dell OptiPlex 3020?",
        "Which pages mention wireless connectivity?",
        "Tell me about the Dell Precision workstation tower",
        "What workstation products are in the catalog?",
    ],
)
def test_a_named_subject_is_never_rewritten(query):
    # These must stay untouched: a wrong guess cannot be allowed to
    # replace what the user actually asked.
    assert needs_rewrite(query) is False


def test_rewrite_rescues_a_question_the_gate_would_block():
    """
    The end-to-end effect that matters: a follow-up with no subject
    of its own becomes searchable.
    """
    from app.retrieval.lexical import tokenize

    follow_up = "Does it support Wi-Fi 6?"
    previous = "Which pages mention wireless connectivity?"

    # On its own, the follow-up shares almost nothing with the index.
    bare_terms = set(tokenize(follow_up))
    assert "wireless" not in bare_terms
    assert "connectivity" not in bare_terms

    result = rewrite_query(
        follow_up, memory(("user", previous))
    )

    assert result.was_rewritten is True
    search_terms = set(tokenize(result.search_query))
    assert {"wireless", "connectivity"} <= search_terms
