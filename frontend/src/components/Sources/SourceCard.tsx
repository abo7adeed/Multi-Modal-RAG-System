import { useState } from 'react'
import type { RAGSource } from '../../services/ragApi'
import { resolveMediaUrl } from '../../services/ragApi'
import { Badge } from '../UI'

interface SourceCardProps {
  source: RAGSource
  // Chunk ids help when several sources come from the same page.
  showChunkId?: boolean
}

type SourceKind = 'image' | 'text' | 'pdf-page' | 'unknown'

function classify(source: RAGSource): SourceKind {
  if (source.content_type === 'image') {
    if (source.page !== null && source.page !== undefined) return 'pdf-page'
    return 'image'
  }
  if (source.content_type === 'text') return 'text'
  return 'unknown'
}

const KIND_LABEL: Record<SourceKind, string> = {
  image: 'Image',
  text: 'Text',
  'pdf-page': 'PDF Page',
  unknown: 'Source',
}

const KIND_STYLE: Record<SourceKind, string> = {
  image: 'border-accent/30',
  text: 'border-line',
  'pdf-page': 'border-accent/30',
  unknown: 'border-line',
}

export function SourceCard({
  source,
  showChunkId = false,
}: SourceCardProps) {
  const [zoomed, setZoomed] = useState(false)
  const kind = classify(source)
  const imageUrl = resolveMediaUrl(source.image_url)

  return (
    <div
      className={`rounded-xl border bg-surface p-3 shadow-soft transition-shadow hover:shadow-card ${KIND_STYLE[kind]}`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          {kind === 'text' ? (
            <Badge variant="neutral">📄 Text</Badge>
          ) : (
            <Badge variant="accent">🖼️ {KIND_LABEL[kind]}</Badge>
          )}
          <span className="truncate text-xs font-medium text-ink">
            {source.source ?? 'Unknown document'}
            {source.page !== null && source.page !== undefined
              ? ` — Page ${source.page}`
              : ''}
          </span>
        </div>
        {source.chunk_id && (showChunkId || source.image_url) && (
          <span
            className="hidden shrink-0 font-mono text-[10px] text-muted sm:block"
            title={source.chunk_id}
          >
            {source.chunk_id.slice(0, 8)}…
          </span>
        )}
      </div>

      {source.snippet && (
        <p className="mt-2 text-xs leading-relaxed text-muted">
          {source.snippet}
        </p>
      )}

      {imageUrl && (
        <div className="mt-2.5">
          <button
            type="button"
            onClick={() => setZoomed(true)}
            className="group relative block w-full overflow-hidden rounded-lg ring-1 ring-line focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
            aria-label="Open larger preview"
          >
            <img
              src={imageUrl}
              alt={
                source.source
                  ? `Image from ${source.source}${source.page !== null ? `, page ${source.page}` : ''}`
                  : 'Retrieved document image'
              }
              loading="lazy"
              className="h-36 w-full object-cover transition-transform duration-200 group-hover:scale-[1.02]"
            />
          </button>
        </div>
      )}

      {zoomed && imageUrl && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Image preview"
          className="fixed inset-0 z-50 flex items-center justify-center bg-canvas/90 p-4 backdrop-blur-sm"
          onClick={() => setZoomed(false)}
          onKeyDown={(event) => {
            if (event.key === 'Escape') setZoomed(false)
          }}
          tabIndex={-1}
        >
          <img
            src={imageUrl}
            alt="Retrieved document image, enlarged"
            className="max-h-[85vh] max-w-full rounded-lg shadow-2xl"
          />
          <button
            type="button"
            onClick={() => setZoomed(false)}
            className="absolute right-4 top-4 rounded-lg bg-ink/10 px-3 py-1.5 text-sm text-ink hover:bg-ink/20"
          >
            Close
          </button>
        </div>
      )}
    </div>
  )
}
