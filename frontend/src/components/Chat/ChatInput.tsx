import { useRef, useState } from 'react'
import type { ChangeEvent, ClipboardEvent, KeyboardEvent } from 'react'
import {
  ACCEPTED_IMAGE_TYPES,
  MAX_ATTACHMENT_BYTES,
  readImageAttachment,
} from '../../services/ragApi'
import type { PendingImage } from '../../services/ragApi'
import { Button, Spinner } from '../UI'

interface ChatInputProps {
  onSend: (query: string, image: PendingImage | null) => void
  isLoading: boolean
}

const ACCEPT_ATTRIBUTE = ACCEPTED_IMAGE_TYPES.join(',')

export function ChatInput({ onSend, isLoading }: ChatInputProps) {
  const [value, setValue] = useState('')
  const [image, setImage] = useState<PendingImage | null>(null)
  const [imageError, setImageError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const canSend = value.trim().length > 0 && !isLoading

  const submit = () => {
    if (!canSend) return
    onSend(value.trim(), image)
    setValue('')
    setImage(null)
    setImageError(null)
    textareaRef.current?.focus()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  const attach = async (file: File | undefined) => {
    if (!file) return

    setImageError(null)

    if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
      setImageError('Attach a JPG, PNG, WEBP or GIF image.')
      return
    }

    if (file.size > MAX_ATTACHMENT_BYTES) {
      setImageError(
        `That image is ${(file.size / 1024 / 1024).toFixed(1)} MB. The limit is 5 MB.`,
      )
      return
    }

    try {
      setImage(await readImageAttachment(file))
    } catch {
      setImageError('That image could not be read.')
    }
  }

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    void attach(event.target.files?.[0])
    // Reset so re-picking the same file fires a change event.
    event.target.value = ''
  }

  const handlePaste = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    const pasted = Array.from(event.clipboardData?.files ?? []).find(
      (file) => file.type.startsWith('image/'),
    )
    if (pasted) {
      event.preventDefault()
      void attach(pasted)
    }
  }

  const removeImage = () => {
    setImage(null)
    setImageError(null)
  }

  return (
    <div className="border-t border-line bg-surface/80 backdrop-blur-md">
      <div className="mx-auto max-w-3xl px-4 py-3 sm:px-6">
        <form
          className="rounded-2xl border border-line bg-surface shadow-card transition-colors focus-within:border-accent/60 focus-within:ring-2 focus-within:ring-accent/20"
          onSubmit={(event) => {
            event.preventDefault()
            submit()
          }}
        >
          {image && (
            <div className="flex items-center gap-3 border-b border-line px-3 py-2.5">
              <img
                src={image.dataUrl}
                alt={`Attached image: ${image.filename}`}
                className="h-14 w-14 rounded-lg object-cover ring-1 ring-line"
              />
              <div className="min-w-0 flex-1">
                <p className="truncate text-xs font-medium text-ink">
                  {image.filename}
                </p>
                <p className="text-[11px] text-muted">
                  Ask a question about this image
                </p>
              </div>
              <button
                type="button"
                onClick={removeImage}
                aria-label="Remove attached image"
                className="rounded-full p-1.5 text-muted transition-colors hover:bg-surface-2 hover:text-ink"
              >
                <svg
                  viewBox="0 0 20 20"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  className="h-4 w-4"
                  aria-hidden="true"
                >
                  <path
                    d="M6 6l8 8M14 6l-8 8"
                    strokeLinecap="round"
                  />
                </svg>
              </button>
            </div>
          )}

          <div className="flex items-end gap-2 p-2">
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPT_ATTRIBUTE}
              onChange={handleFileChange}
              className="hidden"
              aria-hidden="true"
              tabIndex={-1}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              aria-label="Attach an image"
              title="Attach an image (or paste one)"
              className="flex h-[40px] w-[40px] shrink-0 items-center justify-center rounded-xl text-muted transition-colors hover:bg-surface-2 hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
            >
              <svg
                viewBox="0 0 20 20"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                className="h-5 w-5"
                aria-hidden="true"
              >
                <path
                  d="M13.5 6.5V5a1.5 1.5 0 0 0-1.5-1.5h-6A1.5 1.5 0 0 0 4.5 5v10A1.5 1.5 0 0 0 6 16.5h3"
                  strokeLinecap="round"
                />
                <path
                  d="M12 10v6m0 0 2.5-2.5M12 16l-2.5-2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>

            <textarea
              id="rag-input"
              ref={textareaRef}
              rows={1}
              value={value}
              onChange={(event) => setValue(event.target.value)}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste}
              placeholder={
                image
                  ? 'What would you like to know about this image?'
                  : 'Ask about your documents, or attach an image…'
              }
              aria-label="Ask about your documents"
              className="max-h-40 min-h-[40px] flex-1 resize-none bg-transparent px-1 py-2 text-sm text-ink placeholder:text-muted focus:outline-none"
              style={{ height: 'auto' }}
            />
            <Button
              type="submit"
              disabled={!canSend}
              aria-label="Send message"
              className="h-[40px] px-4"
            >
              {isLoading ? <Spinner /> : 'Send'}
            </Button>
          </div>
        </form>

        {imageError && (
          <p
            role="alert"
            className="mt-1.5 text-center text-[11px] text-danger-text"
          >
            {imageError}
          </p>
        )}

        <p className="mt-1.5 hidden text-center text-[11px] text-muted sm:block">
          Enter to send · Shift+Enter for a new line · Paste or attach an image
        </p>
      </div>
    </div>
  )
}
