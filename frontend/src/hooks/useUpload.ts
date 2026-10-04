import { useCallback, useEffect, useRef, useState } from 'react'
import { uploadDocument } from '../services/ragApi'

export type UploadStatus = 'idle' | 'uploading' | 'success' | 'error'

export interface UploadResult {
  filename: string
  chunks_indexed: number
  document_type: string
}

export interface UploadState {
  status: UploadStatus
  progress: number
  error: string | null
  result: UploadResult | null
}

const INITIAL_STATE: UploadState = {
  status: 'idle',
  progress: 0,
  error: null,
  result: null,
}

export const SUPPORTED_EXTENSIONS = ['.pdf', '.jpg', '.jpeg', '.png', '.webp']
export const MAX_FILE_SIZE_MB = 25

export interface UseUploadOptions {
  autoResetMs?: number
}

export function useUpload({ autoResetMs = 6000 }: UseUploadOptions = {}) {
  const [state, setState] = useState<UploadState>(INITIAL_STATE)
  const resetTimer = useRef<number | null>(null)

  useEffect(() => {
    return () => {
      if (resetTimer.current !== null) {
        window.clearTimeout(resetTimer.current)
      }
    }
  }, [])

  const scheduleReset = useCallback(() => {
    if (resetTimer.current !== null) {
      window.clearTimeout(resetTimer.current)
    }
    resetTimer.current = window.setTimeout(() => {
      setState(INITIAL_STATE)
    }, autoResetMs)
  }, [autoResetMs])

  const validate = useCallback((file: File): string | null => {
    const lower = file.name.toLowerCase()
    const okExtension = SUPPORTED_EXTENSIONS.some((ext) =>
      lower.endsWith(ext),
    )
    if (!okExtension) {
      return `Unsupported format. Allowed: ${SUPPORTED_EXTENSIONS.join(', ')}`
    }
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      return `File exceeds the ${MAX_FILE_SIZE_MB} MB limit.`
    }
    if (file.size === 0) {
      return 'The file is empty.'
    }
    return null
  }, [])

  const upload = useCallback(
    async (file: File) => {
      const validationError = validate(file)
      if (validationError) {
        setState({
          status: 'error',
          progress: 0,
          error: validationError,
          result: null,
        })
        scheduleReset()
        return false
      }

      setState({
        status: 'uploading',
        progress: 0,
        error: null,
        result: null,
      })

      try {
        const response = await uploadDocument(file, (percent) => {
          setState((prev) =>
            prev.status === 'uploading' ? { ...prev, progress: percent } : prev,
          )
        })
        setState({
          status: 'success',
          progress: 100,
          error: null,
          result: {
            filename: response.filename,
            chunks_indexed: response.chunks_indexed,
            document_type: response.document_type,
          },
        })
        scheduleReset()
        return true
      } catch (err) {
        setState({
          status: 'error',
          progress: 0,
          error:
            err instanceof Error
              ? err.message
              : 'Upload failed. Please try again.',
          result: null,
        })
        scheduleReset()
        return false
      }
    },
    [scheduleReset, validate],
  )

  const reset = useCallback(() => setState(INITIAL_STATE), [])

  return { state, upload, reset }
}
