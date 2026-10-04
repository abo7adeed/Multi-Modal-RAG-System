export interface RAGSource {
  chunk_id: string | null
  page: number | null
  content_type: string | null
  source: string | null
  image_path: string | null
  image_url: string | null
  snippet: string | null
}

export interface RAGQueryResponse {
  answer: string
  sources: RAGSource[]
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
): Promise<RAGQueryResponse> {
  let response: Response
  try {
    response = await fetch(buildUrl('/api/v1/rag/query'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        ...(image ? { image } : {}),
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
