import { useCallback, useEffect, useRef, useState } from 'react'
import { queryRAGStream, RAGApiError } from '../services/ragApi'
import type {
  PendingImage,
  RAGHistoryTurn,
  RAGQueryResponse,
  RAGSource,
} from '../services/ragApi'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: RAGQueryResponse['sources']
  error?: string
  /** Image the user attached to this message (user messages only). */
  image?: PendingImage | null
  /**
   * True while tokens are still arriving. Drives the typing
   * indicator, and lets the UI show a partial answer as provisional.
   */
  isStreaming?: boolean
}

const WELCOME_MESSAGE: ChatMessage = {
  id: 'welcome',
  role: 'assistant',
  content:
    'Ask me anything about your indexed documents, or attach a picture and ask about it directly. I will ground every answer in the retrieved text and images and show you exactly where each answer came from.',
}

let messageCounter = 0
function nextMessageId(): string {
  messageCounter += 1
  return `msg-${messageCounter}`
}

/**
 * Turns of conversation sent with a request.
 *
 * Mirrors the server's MAX_HISTORY_TURNS. The server keeps no session
 * state, so this window is what lets a follow-up question ("what
 * about its warranty?") be resolved against what came before.
 */
const HISTORY_TURNS = 6

/**
 * The recent conversation, oldest first.
 *
 * Error messages, the welcome message and the in-progress bubble are
 * excluded: they are UI state, not something the user said, and
 * replaying a failure would teach the model that the answer to a
 * question is an error.
 */
function buildHistory(messages: ChatMessage[]): RAGHistoryTurn[] {
  return messages
    .filter(
      (message) =>
        message.id !== 'welcome' &&
        !message.error &&
        !message.isStreaming &&
        message.content.trim().length > 0,
    )
    .slice(-HISTORY_TURNS)
    .map((message) => ({
      role: message.role,
      content: message.content,
    }))
}

/**
 * Replace one message in place, matched by id.
 *
 * Streaming appends a token many times per second; rebuilding the
 * whole array by index would be wrong as soon as a second message
 * were added concurrently, and matching by id keeps the update
 * anchored to the message actually being written.
 */
function patchMessage(
  messages: ChatMessage[],
  id: string,
  patch: Partial<ChatMessage>,
): ChatMessage[] {
  return messages.map((message) =>
    message.id === id ? { ...message, ...patch } : message,
  )
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE])
  const [isLoading, setIsLoading] = useState(false)
  const [lastFailedQuery, setLastFailedQuery] = useState<string | null>(null)
  // The attachment is kept so a retry re-sends the same picture, not
  // a question about a picture that is no longer there.
  const [lastFailedImage, setLastFailedImage] = useState<PendingImage | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    return () => abortRef.current?.abort()
  }, [])

  const ask = useCallback(
    async (query: string, image?: PendingImage | null) => {
      const trimmed = query.trim()
      if (!trimmed || isLoading) return

      setLastFailedQuery(null)
      setLastFailedImage(null)

      // Captured before the new messages are appended, so the history
      // describes the conversation that preceded this question.
      const history = buildHistory(messages)

      setMessages((prev) => [
        ...prev,
        { id: nextMessageId(), role: 'user', content: trimmed, image: image ?? null },
      ])
      setIsLoading(true)

      const controller = new AbortController()
      abortRef.current = controller

      // The assistant message is added up front, empty, so tokens can
      // be appended to it as they arrive. Adding it per token would
      // make the message reorder itself on every chunk.
      const assistantId = nextMessageId()
      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: 'assistant', content: '', isStreaming: true },
      ])

      try {
        await queryRAGStream(
          trimmed,
          image,
          {
            onSources: (sources: RAGSource[]) => {
              setMessages((prev) => patchMessage(prev, assistantId, { sources }))
            },
            onToken: (text: string) => {
              setMessages((prev) => {
                const target = prev.find((message) => message.id === assistantId)
                if (!target) return prev
                return patchMessage(prev, assistantId, {
                  content: target.content + text,
                })
              })
            },
            onDone: () => {
              setMessages((prev) => patchMessage(prev, assistantId, { isStreaming: false }))
            },
          },
          controller.signal,
          undefined,
          history,
        )
      } catch (err) {
        if (controller.signal.aborted) {
          // The user stopped this request; drop the placeholder
          // rather than leaving an empty bubble behind.
          setMessages((prev) => prev.filter((message) => message.id !== assistantId))
          return
        }

        const message =
          err instanceof RAGApiError
            ? err.message
            : 'Something went wrong. Please try again.'

        setMessages((prev) => {
          const target = prev.find((entry) => entry.id === assistantId)

          // Tokens already rendered: keep them and flag the message
          // as failed rather than discarding real content.
          if (target?.content) {
            return patchMessage(prev, assistantId, {
              isStreaming: false,
              error: message,
            })
          }

          // Nothing was written, so the placeholder becomes the error
          // bubble. Dropping it silently would leave the user with a
          // question and no explanation.
          return prev.flatMap((entry) =>
            entry.id === assistantId
              ? [{ ...entry, content: '', isStreaming: false, error: message }]
              : [entry],
          )
        })

        setLastFailedQuery(trimmed)
        setLastFailedImage(image ?? null)
      } finally {
        setIsLoading(false)
        abortRef.current = null
      }
    },
    [isLoading, messages],
  )

  const retry = useCallback(() => {
    if (lastFailedQuery) {
      setMessages((prev) => {
        const withoutError = prev.filter((message) => !message.error)
        return withoutError
      })
      void ask(lastFailedQuery, lastFailedImage)
    }
  }, [ask, lastFailedQuery, lastFailedImage])

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const clear = useCallback(() => {
    abortRef.current?.abort()
    setMessages([WELCOME_MESSAGE])
    setLastFailedQuery(null)
    setLastFailedImage(null)
    setIsLoading(false)
  }, [])

  return {
    messages,
    isLoading,
    lastFailedQuery,
    ask,
    retry,
    stop,
    clear,
  }
}