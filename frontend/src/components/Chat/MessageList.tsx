import type { ChatMessage } from '../../hooks/useChat'
import { useAutoScroll } from '../../hooks/useAutoScroll'
import { MessageBubble } from './MessageBubble'

interface MessageListProps {
  messages: ChatMessage[]
  isLoading: boolean
  onRetry: () => void
  onSuggestion: (text: string) => void
}

// Each suggestion must be answerable from the indexed catalog.
// Verified against the document: "optiplex 3020" appears on pages
// 2/12/13, "workstation" on 2/10/11/17/18/19, "wireless" on
// 3/4/5/7/8/9/11/12.
const SUGGESTIONS = [
  'What is the Dell OptiPlex 3020?',
  'Which pages mention wireless connectivity?',
  'What workstation products are in the catalog?',
]

function EmptyState({ onSuggestion }: { onSuggestion: (text: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-accent-soft text-2xl ring-1 ring-accent/20">
        <span aria-hidden="true">🗂️</span>
      </div>
      <h2 className="text-lg font-semibold tracking-tight text-ink">
        Ask your catalog anything
      </h2>
      <p className="mt-2 max-w-md text-sm leading-relaxed text-muted">
        Every answer is grounded in retrieved text and page images from
        your indexed documents — with sources you can inspect. You can
        also attach a picture and ask about it directly.
      </p>
      <div className="mt-7 flex flex-wrap justify-center gap-2">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => onSuggestion(suggestion)}
            className="rounded-full border border-line bg-surface px-4 py-1.5 text-xs text-muted shadow-soft transition-colors hover:border-accent/40 hover:bg-accent-soft hover:text-accent-text"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-1.5 rounded-2xl rounded-bl-md border border-line bg-surface px-4 py-3.5 shadow-soft">
        {[0, 150, 300].map((delay) => (
          <span
            key={delay}
            className="h-2 w-2 animate-bounce rounded-full bg-muted"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </div>
    </div>
  )
}

export function MessageList({
  messages,
  isLoading,
  onRetry,
  onSuggestion,
}: MessageListProps) {
  const scrollRef = useAutoScroll<HTMLDivElement>(
    `${messages.length}:${isLoading}`,
  )

  const visible = messages.filter((message) => message.id !== 'welcome')

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto scroll-smooth"
      aria-live="polite"
    >
      <div className="mx-auto max-w-3xl space-y-5 px-4 py-6 sm:px-6">
        {visible.length === 0 && !isLoading && (
          <EmptyState onSuggestion={onSuggestion} />
        )}
        {visible.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            onRetry={
              message.error && message === visible[visible.length - 1]
                ? onRetry
                : undefined
            }
          />
        ))}
        {isLoading && <TypingIndicator />}
      </div>
    </div>
  )
}
