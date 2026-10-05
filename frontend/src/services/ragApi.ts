export interface RAGSource {
  chunk_id: string | null
  page: number | null
  content_type: string | null
  source: string | null
  image_path: string | null
  image_url: string | null
  snippet: string | null
  /** Retrieval rank score, for ordering/labelling in the UI. */
  score: number | null
  /** "document" for an indexed chunk; "visual_match" is never cited. */
  kind: string
}

export interface RAGQueryResponse {
  answer: string
  sources: RAGSource[]
}

/**
 * One earlier message, so the server can resolve follow-up questions
 * like "what about its warranty?".
 *
 * The client owns the transcript; the API keeps no session state.
 */
export interface RAGHistoryTurn {
  role: 'user' | 'assistant'
  content: string
}

/**
 * An image the user attached to their question, as the API expects
 * it: raw base64 without a data: URL prefix.
 */
export interface RAGImageAttachment {
  filename: string
  media_type: string
  data: string
}

/** An attachment plus the data URL used to preview it locally. */
export interface PendingImage extends RAGImageAttachment {
  dataUrl: string
}

/**
 * Mirrors API_MAX_QUERY_IMAGE_SIZE_MB. Checked here so an oversized
 * file is refused before it is read into memory and sent.
 */
export const MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024

export const ACCEPTED_IMAGE_TYPES = [
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/gif',
]

export interface DocumentUploadResponse {
  document_id: string
  filename: string
  document_type: string
  chunks_indexed: number
}

export interface ApiErrorPayload {
  message: string
  type: string
}

export interface ApiErrorResponse {
  error: ApiErrorPayload
}

export type RAGErrorType =
  | 'rate_limit'
  | 'timeout'
  | 'retrieval'
  | 'invalid_request'
  | 'server'
  | 'network'
  | 'unknown'

export class RAGApiError extends Error {
  readonly errorType: RAGErrorType
  readonly status: number | null

  constructor(message: string, errorType: RAGErrorType, status: number | null) {
    super(message)
    this.name = 'RAGApiError'
    this.errorType = errorType
    this.status = status
  }
}

function mapErrorType(type: string, status: number | null): RAGErrorType {
  switch (type) {
    case 'ProviderRateLimitError':
      return 'rate_limit'
    case 'ProviderTimeoutError':
      return 'timeout'
    case 'RetrievalError':
      return 'retrieval'
    case 'InvalidRequestError':
      return 'invalid_request'
    default:
      if (status === 429) return 'rate_limit'
      if (status === 504) return 'timeout'
      if (status === 400) return 'invalid_request'
      if (status !== null && status >= 500) return 'server'
      return 'unknown'
  }
}

async function parseError(response: Response): Promise<RAGApiError> {
  let message = 'Something went wrong. Please try again.'
  let type = 'Unknown'

  try {
    const body = (await response.json()) as Partial<ApiErrorResponse> | null
    if (body?.error?.message) {
      message = body.error.message
      type = body.error.type ?? 'Unknown'
    }
  } catch {
    // Non-JSON error body; keep defaults.
  }

  return new RAGApiError(
    message,
    mapErrorType(type, response.status),
    response.status,
  )
}

function toApiError(err: unknown): RAGApiError {
  if (err instanceof RAGApiError) return err
  if (err instanceof Error) {
    return new RAGApiError(err.message, 'network', null)
  }
  return new RAGApiError('Network error.', 'network', null)
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

export const API_BASE = API_BASE_URL

function buildUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}

function resolveMediaUrl(url: string | null): string | null {
  if (!url) return null
  if (/^https?:\/\//i.test(url)) return url
  return `${API_BASE_URL}${url}`
}

export async function queryRAG(
  query: string,
  image?: RAGImageAttachment | null,
  signal?: AbortSignal,
  maxSources?: number,
  history?: RAGHistoryTurn[],
): Promise<RAGQueryResponse> {
  let response: Response
  try {
    response = await fetch(buildUrl('/api/v1/rag/query'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        ...(image ? { image } : {}),
        ...(maxSources ? { max_sources: maxSources } : {}),
        ...(history?.length ? { history } : {}),
      }),
      signal,
    })
  } catch (err) {
    throw toApiError(err)
  }

  if (!response.ok) {
    throw await parseError(response)
  }

  try {
    return (await response.json()) as RAGQueryResponse
  } catch (err) {
    throw toApiError(err)
  }
}

// ------------------------------------------------------------
// Streaming (Server-Sent Events)
// ------------------------------------------------------------

/**
 * One decoded SSE frame. Mirrors the server's RAGStreamEvent plus a
 * dedicated `error` event for failures that arrive after the
 * response has already started.
 */
export type RAGStreamFrame =
  | { type: 'sources'; sources: RAGSource[] }
  | { type: 'token'; text: string }
  | { type: 'done' }
  | { type: 'error'; message: string; errorType: string }

export interface RAGStreamHandlers {
  /** Citations, sent before any token so they can render early. */
  onSources?: (sources: RAGSource[]) => void
  /** Called per token; append the text to what you already have. */
  onToken?: (text: string) => void
  /** Called once when the answer completes successfully. */
  onDone?: () => void
}

/**
 * Decode a partial SSE buffer into complete frames.
 *
 * Exported for testing: chunk boundaries from the network fall
 * anywhere, including mid-frame, so the leftover must be carried
 * over rather than discarded.
 */
export function parseSSEFrames(buffer: string): { frames: RAGStreamFrame[]; rest: string } {
  const frames: RAGStreamFrame[] = []

  // Frames are separated by a blank line. Only complete frames are
  // consumed; anything after the last separator is the next frame's
  // partial text.
  const parts = buffer.split('\n\n')
  const rest = parts.pop() ?? ''

  for (const part of parts) {
    const frame = parseSSEFrame(part)
    if (frame) frames.push(frame)
  }

  return { frames, rest }
}

function parseSSEFrame(raw: string): RAGStreamFrame | null {
  const lines = raw.split('\n')
  let event: string | null = null
  const dataLines: string[] = []

  for (const line of lines) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trim())
    }
  }

  if (!event || dataLines.length === 0) return null

  let payload: Record<string, unknown>
  try {
    payload = JSON.parse(dataLines.join('\n')) as Record<string, unknown>
  } catch {
    // A malformed frame is skipped rather than thrown: one bad frame
    // should not lose the rest of a working answer.
    return null
  }

  switch (event) {
    case 'sources':
      return { type: 'sources', sources: (payload.sources ?? []) as RAGSource[] }
    case 'token':
      return { type: 'token', text: String(payload.text ?? '') }
    case 'done':
      return { type: 'done' }
    case 'error':
      return {
        type: 'error',
        message: String(payload.message ?? 'Something went wrong. Please try again.'),
        errorType: String(payload.type ?? 'Unknown'),
      }
    default:
      // Unknown event names are ignored so the server can add
      // events without breaking older clients.
      return null
  }
}

/**
 * Ask a question and receive the answer as it is generated.
 *
 * Resolves once the stream completes. Throws if the request fails
 * before the response starts (a real HTTP status), or reports an
 * `error` frame via onError when generation fails mid-stream.
 */
export function queryRAGStream(
  query: string,
  image: RAGImageAttachment | null | undefined,
  handlers: RAGStreamHandlers,
  signal?: AbortSignal,
  maxSources?: number,
  history?: RAGHistoryTurn[],
): Promise<void> {
  return (async () => {
    let response: Response
    try {
      response = await fetch(buildUrl('/api/v1/rag/stream'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({
          query,
          ...(image ? { image } : {}),
          ...(maxSources ? { max_sources: maxSources } : {}),
          ...(history?.length ? { history } : {}),
        }),
        signal,
      })
    } catch (err) {
      throw toApiError(err)
    }

    // A pre-stream failure still carries a real HTTP status, so it is
    // reported the same way as the buffered route.
    if (!response.ok) {
      throw await parseError(response)
    }

    if (!response.body) {
      throw new RAGApiError(
        'Streaming is not supported by this browser.',
        'unknown',
        null,
      )
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let sawDone = false

    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break

        // stream:true keeps multi-byte characters intact across
        // chunk boundaries, which matters for non-Latin answers.
        buffer += decoder.decode(value, { stream: true })

        const { frames, rest } = parseSSEFrames(buffer)
        buffer = rest

        for (const frame of frames) {
          if (frame.type === 'sources') {
            handlers.onSources?.(frame.sources)
          } else if (frame.type === 'token') {
            handlers.onToken?.(frame.text)
          } else if (frame.type === 'done') {
            sawDone = true
          } else if (frame.type === 'error') {
            throw new RAGApiError(
              frame.message,
              mapErrorType(frame.errorType, null),
              null,
            )
          }
        }
      }
    } catch (err) {
      if (signal?.aborted) {
        // An abort is a user action, not a failure; let the caller
        // decide what to show.
        throw err
      }
      throw toApiError(err)
    } finally {
      reader.releaseLock()
    }

    // `done` is the only completion signal. A stream that stops
    // without it was truncated, and reporting success would show a
    // half-finished answer as if it were complete.
    if (!sawDone) {
      throw new RAGApiError(
        'The answer was cut off before it finished. Please try again.',
        'network',
        null,
      )
    }

    handlers.onDone?.()
  })()
}

export async function uploadDocument(
  file: File,
  onProgress?: (percent: number) => void,
  signal?: AbortSignal,
): Promise<DocumentUploadResponse> {
  // XMLHttpRequest gives real upload progress events; fetch does not.
  return new Promise<DocumentUploadResponse>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', buildUrl('/api/v1/documents'))

    xhr.upload.onprogress = (event) => {
      if (onProgress && event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100))
      }
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as DocumentUploadResponse)
        } catch {
          reject(new RAGApiError('Invalid server response.', 'server', xhr.status))
        }
      } else {
        let message = 'Upload failed. Please try again.'
        let type = 'Unknown'
        try {
          const body = JSON.parse(xhr.responseText)
          if (typeof body?.detail === 'string') {
            message = body.detail
            type = 'InvalidRequestError'
          } else if (body?.error?.message) {
            message = body.error.message
            type = body.error.type
          }
        } catch {
          // keep defaults
        }
        reject(new RAGApiError(message, mapErrorType(type, xhr.status), xhr.status))
      }
    }

    xhr.onerror = () => reject(new RAGApiError('Network error during upload.', 'network', null))
    signal?.addEventListener('abort', () => {
      xhr.abort()
      reject(new RAGApiError('Upload cancelled.', 'invalid_request', null))
    })

    const formData = new FormData()
    formData.append('file', file)
    xhr.send(formData)
  })
}

/**
 * Read an image file into a sendable attachment plus a preview URL.
 *
 * One read produces both: the data URL doubles as the base64 payload
 * (minus the prefix) the API expects, so the bytes are never encoded
 * twice and the preview needs no object URL to revoke.
 */
export function readImageAttachment(file: File): Promise<PendingImage> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.onerror = () => {
      reject(new RAGApiError('The image could not be read.', 'invalid_request', null))
    }

    reader.onload = () => {
      const dataUrl = String(reader.result ?? '')
      const comma = dataUrl.indexOf(',')
      if (comma < 0) {
        reject(new RAGApiError('The image could not be read.', 'invalid_request', null))
        return
      }

      resolve({
        filename: file.name || 'attachment',
        media_type: file.type || 'image/jpeg',
        data: dataUrl.slice(comma + 1),
        dataUrl,
      })
    }

    reader.readAsDataURL(file)
  })
}

export { resolveMediaUrl }
