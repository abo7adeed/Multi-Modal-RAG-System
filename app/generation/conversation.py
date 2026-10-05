"""
Conversation memory and query rewriting.

Follow-up questions are the hard case for a lexical retrieval gate:
"What about its warranty?" shares no searchable words with the
index, so it is either blocked as off-topic or answered from
whatever happens to rank highest. The referent ("its") only exists in
an earlier turn.

This module resolves that reference *before* retrieval, so the search
sees a standalone query. The rewrite is deliberately rule-based and
deterministic: it costs nothing, adds no latency, never invents a
subject, and - unlike an LLM rewriter - is fully testable.

Rewriting only ever *adds* the earlier question's words. The user's
own question is never rewritten away, and a query that already names
its subject is left untouched, so a wrong guess cannot silently
replace what the user actually asked.
"""
from typing import Literal

from pydantic import BaseModel, Field

import re

from app.retrieval.lexical import tokenize

#: Turns kept from the client's history. Enough to follow a couple of
#: follow-ups without letting an unbounded client-supplied list grow
#: the prompt.
MAX_HISTORY_TURNS = 6

#: Characters of history text passed to the model. Bounds the prompt
#: against a client that sends very long turns.
MAX_HISTORY_CHARS = 2000

#: Matches words without dropping stopwords. `tokenize` is tuned for
#: the lexical index and removes exactly the pronouns this module
#: needs to see, so pronoun detection uses its own pattern.
_WORD_PATTERN = re.compile(r"[a-z0-9]+")

#: A query with at most this many searchable words is treated as
#: possibly referring to earlier context. "What about its warranty?"
#: tokenizes to ["warranty"] - one word, and the subject is missing.
#: "What is the Dell OptiPlex 3020 warranty?" has three and names its
#: own subject.
MIN_SELF_CONTAINED_TERMS = 3

#: Openers that signal a question continuing from the previous turn.
#: Matched against the lowercased query after stripping punctuation.
CONTINUATION_OPENERS = frozenset(
    {
        "and",
        "also",
        "anyway",
        "but",
        "does",
        "doesnt",
        "how",
        "is",
        "it",
        "its",
        "more",
        "ok",
        "okay",
        "or",
        "so",
        "tell",
        "that",
        "the",
        "then",
        "there",
        "they",
        "this",
        "those",
        "what",
        "whats",
        "when",
        "where",
        "which",
        "who",
        "why",
        "yes",
    }
)

#: Pronouns and demonstratives that point outside the query.
#:
#: These are the strongest follow-up signal available, and they are
#: checked before the word-count heuristic: tokenizing splits
#: "Wi-Fi" into "wi" and "fi", which inflates the count of a query
#: like "Does it support Wi-Fi 6?" to three words even though its
#: subject ("it") is entirely missing. A pronoun has no referent the
#: retriever can resolve, so its presence means the query is not
#: self-contained.
PRONOUNS = frozenset(
    {
        "he",
        "her",
        "him",
        "his",
        "it",
        "its",
        "she",
        "that",
        "their",
        "them",
        "these",
        "they",
        "this",
        "those",
    }
)

#: Phrases that carry no searchable subject of their own.
VAGUE_PHRASES = (
    "tell me more",
    "more information",
    "more info",
    "what about that",
    "what about it",
    "anything else",
    "go on",
    "keep going",
    "and then",
)


class ConversationTurn(BaseModel):
    """One prior message in the conversation."""

    role: Literal["user", "assistant"]
    content: str


class ConversationMemory:
    """
    Bounded view of the conversation so far.

    The client owns the transcript and sends the recent turns with
    each request; the server keeps no per-session state. That keeps
    the API stateless and horizontally scalable, at the cost of
    trusting the client's window - which is fine, because the history
    only ever influences retrieval, never the answer's grounding.
    """

    def __init__(
        self,
        turns: list[ConversationTurn] | None = None,
        max_turns: int = MAX_HISTORY_TURNS,
        max_chars: int = MAX_HISTORY_CHARS,
    ):
        self.max_chars = max_chars
        self.turns = (turns or [])[-max_turns:]

    def __bool__(self) -> bool:
        return bool(self.turns)

    def recent(self) -> list[ConversationTurn]:
        """The retained turns, oldest first."""
        return list(self.turns)

    @property
    def previous_user_query(self) -> str | None:
        """
        The most recent user question before this one.

        This is the referent for a follow-up, so it is the only turn
        used for rewriting. Assistant answers are deliberately *not*
        used: an answer is model-generated text, and folding it into
        the search query would let the model bias its own retrieval.
        """
        for turn in reversed(self.turns):
            if turn.role == "user" and turn.content.strip():
                return turn.content.strip()
        return None

    def as_prompt_context(self) -> str | None:
        """
        Render the history for the model.

        None when there is nothing worth sending, so the caller can
        omit the section entirely rather than adding an empty header.
        """
        lines = [
            f"{turn.role.capitalize()}: {turn.content.strip()}"
            for turn in self.turns
            if turn.content.strip()
        ]
        if not lines:
            return None

        text = "\n".join(lines)
        if len(text) > self.max_chars:
            # Keep the tail: the most recent turns are the ones a
            # follow-up refers to.
            text = "…" + text[-self.max_chars :]

        return text


class RewrittenQuery(BaseModel):
    """
    The outcome of resolving a follow-up question.

    `search_query` is what retrieval uses; `query` stays the user's
    own words so the model answers what was actually asked rather
    than the rewritten form.
    """

    query: str
    search_query: str
    was_rewritten: bool = False
    #: Short tag for logs and tests, e.g. "follow_up".
    reason: str | None = None


def _has_pronoun(query: str) -> bool:
    """Whether the query uses a word that refers outside itself."""
    words = _WORD_PATTERN.findall(query.lower())
    return any(word in PRONOUNS for word in words)


def _is_vague(query: str) -> bool:
    lowered = " ".join(query.lower().split())
    return any(phrase in lowered for phrase in VAGUE_PHRASES)


def needs_rewrite(query: str) -> bool:
    """
    Whether a query probably depends on earlier context.

    Signals, in order of confidence:
      - it contains a vague phrase ("tell me more");
      - it contains a pronoun with no antecedent of its own;
      - it has too few searchable words to identify a subject.

    A query that names its own subject is left alone, so a wrong
    guess can never silently replace what the user asked.
    """
    if _is_vague(query):
        return True

    terms = tokenize(query)

    # A pronoun cannot be resolved by the retriever, whatever else
    # the query contains. Checked on raw words, not on `terms`,
    # because the index tokenizer drops pronouns as stopwords.
    if _has_pronoun(query):
        return True

    if len(terms) >= MIN_SELF_CONTAINED_TERMS:
        # It names its own subject; rewriting would only dilute it.
        return False

    if not terms:
        return True

    first_word = " ".join(query.lower().split()).split(" ", 1)[0]
    return first_word.strip("?.!,") in CONTINUATION_OPENERS


def rewrite_query(
    query: str,
    memory: ConversationMemory | None,
) -> RewrittenQuery:
    """
    Resolve a follow-up question into a standalone search query.

    The previous *question* is prepended for retrieval only. The
    user's wording is preserved for the model, and the earlier turns
    are handed to it separately as context.
    """
    if memory is None or not memory:
        return RewrittenQuery(query=query, search_query=query)

    previous = memory.previous_user_query
    if not previous:
        return RewrittenQuery(query=query, search_query=query)

    if not needs_rewrite(query):
        return RewrittenQuery(query=query, search_query=query)

    # The previous question first so its subject terms carry the most
    # weight in the lexical index; the new question still dominates
    # any term both share, because BM25 counts repeats.
    search_query = f"{previous} {query}".strip()

    return RewrittenQuery(
        query=query,
        search_query=search_query,
        was_rewritten=True,
        reason="follow_up",
    )


__all__ = [
    "ConversationMemory",
    "ConversationTurn",
    "MAX_HISTORY_TURNS",
    "RewrittenQuery",
    "needs_rewrite",
    "rewrite_query",
]