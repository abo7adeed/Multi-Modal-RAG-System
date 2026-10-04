import { Badge, Button } from '../UI'

export type BackendStatus = 'checking' | 'online' | 'offline'

interface HeaderProps {
  status: BackendStatus
  onUploadClick: () => void
  onToggleTheme: () => void
  themeMode: 'light' | 'dark' | 'system'
}

const THEME_ICON: Record<HeaderProps['themeMode'], string> = {
  light: '☀️',
  dark: '🌙',
  system: '🖥️',
}

export function Header({
  status,
  onUploadClick,
  onToggleTheme,
  themeMode,
}: HeaderProps) {
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-surface/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-5xl items-center justify-between gap-3 px-4 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent text-sm font-bold text-accent-ink shadow-soft">
            R
          </div>
          <div className="leading-tight">
            <h1 className="text-sm font-semibold tracking-tight text-ink">
              MultiModal RAG
            </h1>
            <p className="hidden text-[11px] text-muted sm:block">
              Grounded answers from your documents
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="hidden sm:inline-flex">
            {status === 'checking' ? (
              <Badge variant="neutral">Checking…</Badge>
            ) : status === 'online' ? (
              <Badge variant="success">
                <span className="h-1.5 w-1.5 rounded-full bg-success-dot" />
                Online
              </Badge>
            ) : (
              <Badge variant="danger">
                <span className="h-1.5 w-1.5 rounded-full bg-danger" />
                Offline
              </Badge>
            )}
          </span>

          <Button variant="secondary" size="sm" onClick={onUploadClick}>
            <span aria-hidden="true">＋</span> Documents
          </Button>

          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleTheme}
            aria-label={`Theme: ${themeMode}. Click to change.`}
            title={`Theme: ${themeMode}`}
          >
            <span aria-hidden="true">{THEME_ICON[themeMode]}</span>
          </Button>
        </div>
      </div>
    </header>
  )
}
