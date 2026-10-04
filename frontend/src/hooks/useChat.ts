import { useCallback, useEffect, useRef, useState } from 'react'
import { queryRAG, RAGApiError } from '../services/ragApi'
import type { PendingImage, RAGQueryResponse } from '../services/ragApi'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: RAGQueryResponse['sources']
  error?: string
  /** Image the user attached to this message (user messages only). */
  image?: PendingImage | null
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
      setMessages((prev) => [
        ...prev,
        { id: nextMessageId(), role: 'user', content: trimmed, image: image ?? null },
      ])
      setIsLoading(true)

      const controller = new AbortController()
      abortRef.current = controller

      try {
        const response = await queryRAG(trimmed, image, controller.signal)
        setMessages((prev) => [
          ...prev,
          {
            id: nextMessageId(),
            role: 'assistant',
            content: response.answer,
            sources: response.sources,
          },
        ])
      } catch (err) {
        const message =
          err instanceof RAGApiError
            ? err.message
            : 'Something went wrong. Please try again.'
        setMessages((prev) => [
          ...prev,
          { id: nextMessageId(), role: 'assistant', content: '', error: message },
        ])
        setLastFailedQuery(trimmed)
        setLastFailedImage(image ?? null)
      } finally {
        setIsLoading(false)
        abortRef.current = null
      }
    },
    [isLoading],
  )

  const retry = useCallback(() => {
    if (lastFailedQuery) {
      setMessages((prev) => {
        const withoutError = [...prev]
        const last = withoutError[withoutError.length - 1]
        if (last?.error) withoutError.pop()
        return withoutError
      })
      void ask(lastFailedQuery, lastFailedImage)
    }
  }, [ask, lastFailedQuery, lastFailedImage])

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
    clear,
  }
}
