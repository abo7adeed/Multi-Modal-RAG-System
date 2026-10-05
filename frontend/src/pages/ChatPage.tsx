import { useCallback, useEffect, useState } from 'react'
import { Header } from '../components/Layout'
import type { BackendStatus } from '../components/Layout'
import { MessageList, ChatInput } from '../components/Chat'
import { UploadPanel } from '../components/Upload'
import { useChat } from '../hooks/useChat'
import { useTheme } from '../hooks/useTheme'
import { API_BASE } from '../services/ragApi'
import type { PendingImage } from '../services/ragApi'

export default function ChatPage() {
  const { messages, isLoading, ask, retry, stop, clear } = useChat()
  const { mode, cycleTheme } = useTheme()
  const [uploadOpen, setUploadOpen] = useState(false)
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking')

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()

    fetch(`${API_BASE}/api/v1/health`, { signal: controller.signal })
      .then((response) => {
        if (!cancelled) {
          setBackendStatus(response.ok ? 'online' : 'offline')
        }
      })
      .catch(() => {
        if (!cancelled) setBackendStatus('offline')
      })

    return () => {
      cancelled = true
      controller.abort()
    }
  }, [])

  const handleSend = useCallback(
    (query: string, image: PendingImage | null) => {
      void ask(query, image)
    },
    [ask],
  )

  return (
    <div className="flex h-dvh flex-col bg-canvas text-ink">
      <Header
        status={backendStatus}
        onUploadClick={() => setUploadOpen(true)}
        onToggleTheme={cycleTheme}
        themeMode={mode}
      />

      <main className="flex min-h-0 flex-1 flex-col">
        <MessageList
          messages={messages}
          isLoading={isLoading}
          onRetry={retry}
          onSuggestion={(text) => handleSend(text, null)}
        />
        <ChatInput onSend={handleSend} isLoading={isLoading} />
      </main>

      {uploadOpen && <UploadPanel onClose={() => setUploadOpen(false)} />}

      {/* A long generation should be interruptible: without this the
          user can only wait out a model that is taking minutes. */}
      {isLoading && (
        <button
          type="button"
          onClick={stop}
          className="fixed bottom-24 left-1/2 -translate-x-1/2 rounded-full border border-line bg-surface/90 px-4 py-1.5 text-xs font-medium text-muted shadow-card backdrop-blur transition-colors hover:bg-surface-2 hover:text-ink"
        >
          Stop generating
        </button>
      )}

      {messages.length > 1 && (
        <button
          type="button"
          onClick={clear}
          className="fixed bottom-24 right-5 rounded-full border border-line bg-surface/90 px-3 py-1.5 text-xs text-muted shadow-card backdrop-blur transition-colors hover:bg-surface-2 hover:text-ink"
        >
          Clear conversation
        </button>
      )}
    </div>
  )
}
