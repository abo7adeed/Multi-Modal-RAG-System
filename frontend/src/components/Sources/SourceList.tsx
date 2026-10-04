import { useState } from 'react'
import type { RAGSource } from '../../services/ragApi'
import { SourceCard } from './SourceCard'

interface SourceListProps {
  sources: RAGSource[]
}

export function SourceList({ sources }: SourceListProps) {
  const [expanded, setExpanded] = useState(true)

  if (sources.length === 0) return null

  return (
    <section
      aria-label="Answer sources"
      className="mt-3 rounded-xl border border-line bg-surface-2/60 p-3"
    >
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
        className="flex w-full items-center justify-between text-left"
      >
        <span className="text-xs font-semibold uppercase tracking-wider text-muted">
          Sources ({sources.length})
        </span>
        <span
          aria-hidden="true"
          className={`text-muted transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
        >
          ▾
        </span>
      </button>

      {expanded && (
        <div
          className={`mt-2.5 grid grid-cols-1 gap-2 ${
            sources.length > 1 ? 'sm:grid-cols-2' : ''
          }`}
        >
          {sources.map((source, index) => (
            <SourceCard
              key={source.chunk_id ?? index}
              source={source}
              showChunkId={sources.length > 1}
            />
          ))}
        </div>
      )}
    </section>
  )
}
