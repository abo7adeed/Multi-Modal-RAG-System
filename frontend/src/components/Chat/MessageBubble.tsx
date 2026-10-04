import { useState } from 'react'
import type { ChatMessage } from '../../hooks/useChat'
import { SourceList } from '../Sources'
import { Button } from '../UI'

interface MessageBubbleProps {
  message: ChatMessage
  onRetry?: () => void
}

export function MessageBubble({ message, onRetry }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const hasError = Boolean(message.error)
  const [zoom, setZoom] = useState(false)

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[92%] sm:max-w-[85%] ${isUser ? 'items-end' : 'items-start'}`}
      >
        {/* The attachment sits outside the bubble: letterboxing a photo
            inside the coloured bubble reads as a rendering glitch. */}
        {message.image && (
          <button
            type="button"
            onClick={() => setZoom(true)}
            className="mb-1.5 block w-full overflow-hidden rounded-2xl shadow-card ring-1 ring-line focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
            aria-label={`Open larger preview of ${message.image.filename}`}
          >
            <img
              src={message.image.dataUrl}
              alt={`Image you attached: ${message.image.filename}`}
              className="max-h-80 w-full bg-surface-2 object-cover"
            />
          </button>
        )}

        <div
          className={`overflow-hidden rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'rounded-br-md bg-accent text-accent-ink shadow-soft'
              : hasError
                ? 'rounded-bl-md border border-danger/30 bg-danger-soft text-danger-text'
                : 'rounded-bl-md border border-line bg-surface text-ink shadow-soft'
          }`}
        >
          {hasError ? (
            <div className="flex flex-col gap-2">
              <p>⚠️ {message.error}</p>
              {onRetry && (
                <Button variant="secondary" size="sm" onClick={onRetry}>
                  Retry
                </Button>
              )}
            </div>
          ) : (
            message.content && (
              <p className="whitespace-pre-wrap">{message.content}</p>
            )
          )}
        </div>

        {!isUser && message.sources && message.sources.length > 0 && (
          <SourceList sources={message.sources} />
        )}
      </div>

      {zoom && message.image && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Attached image"
          className="fixed inset-0 z-50 flex items-center justify-center bg-canvas/90 p-4 backdrop-blur-sm"
          onClick={() => setZoom(false)}
          onKeyDown={(event) => {
            if (event.key === 'Escape') setZoom(false)
          }}
          tabIndex={-1}
        >
          <img
            src={message.image.dataUrl}
            alt={`Image you attached: ${message.image.filename}, enlarged`}
            className="max-h-[85vh] max-w-full rounded-lg shadow-2xl"
          />
          <button
            type="button"
            onClick={() => setZoom(false)}
            className="absolute right-4 top-4 rounded-lg bg-ink/10 px-3 py-1.5 text-sm text-ink hover:bg-ink/20"
          >
            Close
          </button>
        </div>
      )}
    </div>
  )
}
