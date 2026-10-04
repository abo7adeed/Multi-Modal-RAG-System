import { useCallback, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { useUpload } from '../../hooks/useUpload'
import { Button, Spinner } from '../UI'

interface UploadPanelProps {
  onClose: () => void
}

export function UploadPanel({ onClose }: UploadPanelProps) {
  const { state, upload } = useUpload()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragActive, setDragActive] = useState(false)

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0]
      if (file) {
        void upload(file)
      }
    },
    [upload],
  )

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)
    handleFiles(event.dataTransfer.files)
  }

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(true)
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Upload documents"
      className="fixed inset-0 z-50 flex items-center justify-center bg-canvas/80 p-4 backdrop-blur-sm"
      onClick={onClose}
      onKeyDown={(event) => {
        if (event.key === 'Escape') onClose()
      }}
      tabIndex={-1}
    >
      <div
        className="w-full max-w-lg rounded-2xl border border-line bg-surface p-6 shadow-card"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold tracking-tight text-ink">
              Add documents
            </h2>
            <p className="mt-0.5 text-sm text-muted">
              PDF or image (JPG, PNG, WEBP) up to 25 MB.
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            aria-label="Close upload dialog"
          >
            ✕
          </Button>
        </div>

        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={() => setDragActive(false)}
          className={`rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
            dragActive
              ? 'border-accent bg-accent-soft'
              : 'border-line hover:border-accent/40'
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.jpg,.jpeg,.png,.webp,application/pdf,image/jpeg,image/png,image/webp"
            className="sr-only"
            onChange={(event) => handleFiles(event.target.files)}
          />

          {state.status === 'uploading' ? (
            <div className="flex flex-col items-center gap-3">
              <Spinner className="h-6 w-6 text-accent" />
              <p className="text-sm text-muted">
                Uploading… {state.progress}%
              </p>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
                <div
                  className="h-full rounded-full bg-accent transition-[width] duration-200"
                  style={{ width: `${state.progress}%` }}
                />
              </div>
            </div>
          ) : state.status === 'success' ? (
            <div className="flex flex-col items-center gap-2">
              <span className="text-2xl" aria-hidden="true">
                ✅
              </span>
              <p className="text-sm font-medium text-success">
                {state.result?.filename} indexed
              </p>
              <p className="text-xs text-muted">
                {state.result?.chunks_indexed} chunks added ·{' '}
                {state.result?.document_type}
              </p>
            </div>
          ) : state.status === 'error' ? (
            <div className="flex flex-col items-center gap-2">
              <span className="text-2xl" aria-hidden="true">
                ⚠️
              </span>
              <p className="text-sm font-medium text-danger-text">
                {state.error}
              </p>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => inputRef.current?.click()}
              >
                Try another file
              </Button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <span className="text-3xl" aria-hidden="true">
                📥
              </span>
              <p className="text-sm text-muted">
                Drag &amp; drop a file here, or
              </p>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => inputRef.current?.click()}
              >
                Browse files
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
