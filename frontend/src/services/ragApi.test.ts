import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  queryRAG,
  readImageAttachment,
  resolveMediaUrl,
  RAGApiError,
  API_BASE,
} from './ragApi'

describe('queryRAG', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(),
    )
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('returns parsed response on success', async () => {
    const mockResponse = {
      answer: 'Dell laptops and desktops.',
      sources: [
        {
          chunk_id: 'text-001',
          page: 4,
          content_type: 'text',
          source: 'dell_catalog.pdf',
          image_path: null,
          image_url: null,
        },
      ],
    }

    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify(mockResponse), { status: 200 }),
    )

    const result = await queryRAG('What Dell products are shown?')

    expect(result.answer).toBe('Dell laptops and desktops.')
    expect(result.sources).toHaveLength(1)
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/rag/query'),
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('maps rate limit errors to rate_limit type', async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(
        JSON.stringify({
          error: {
            message: 'Generation service is temporarily rate limited.',
            type: 'ProviderRateLimitError',
          },
        }),
        { status: 429 },
      ),
    )

    await expect(queryRAG('test')).rejects.toMatchObject({
      errorType: 'rate_limit',
      status: 429,
    })
  })

  it('maps timeout errors to timeout type', async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(
        JSON.stringify({
          error: {
            message: 'Generation took too long to respond.',
            type: 'ProviderTimeoutError',
          },
        }),
        { status: 504 },
      ),
    )

    await expect(queryRAG('test')).rejects.toBeInstanceOf(RAGApiError)
    await expect(queryRAG('test')).rejects.toMatchObject({
      errorType: 'timeout',
    })
  })

  it('maps network failures to network type', async () => {
    vi.mocked(fetch).mockRejectedValue(new TypeError('Failed to fetch'))

    await expect(queryRAG('test')).rejects.toMatchObject({
      errorType: 'network',
    })
  })

  it('handles non-JSON error bodies gracefully', async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response('Internal Server Error', { status: 500 }),
    )

    await expect(queryRAG('test')).rejects.toMatchObject({
      errorType: 'server',
      message: 'Something went wrong. Please try again.',
    })
  })

  it('omits the image field when none is attached', async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ answer: 'ok', sources: [] }), {
        status: 200,
      }),
    )

    await queryRAG('What is the OptiPlex?')

    const body = JSON.parse(
      vi.mocked(fetch).mock.calls[0][1]!.body as string,
    )
    expect(body).toEqual({ query: 'What is the OptiPlex?' })
  })

  it('sends an attached image alongside the question', async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ answer: 'A red square.', sources: [] }), {
        status: 200,
      }),
    )

    const image = {
      filename: 'photo.png',
      media_type: 'image/png',
      data: 'aGVsbG8=',
    }

    await queryRAG('What is this?', image)

    const body = JSON.parse(
      vi.mocked(fetch).mock.calls[0][1]!.body as string,
    )
    expect(body.query).toBe('What is this?')
    expect(body.image).toEqual(image)
  })
})

describe('readImageAttachment', () => {
  it('strips the data URL prefix from the base64 payload', async () => {
    const file = new File(['hello'], 'photo.png', {
      type: 'image/png',
    })

    const attachment = await readImageAttachment(file)

    expect(attachment.filename).toBe('photo.png')
    expect(attachment.media_type).toBe('image/png')
    // Raw base64 only: the API does not accept a data: URL.
    expect(attachment.data).toBe(btoa('hello'))
    expect(attachment.dataUrl).toBe(`data:image/png;base64,${btoa('hello')}`)
  })

  it('falls back to a generic media type and filename', async () => {
    const file = new File(['hello'], '', { type: '' })

    const attachment = await readImageAttachment(file)

    expect(attachment.filename).toBe('attachment')
    expect(attachment.media_type).toBe('image/jpeg')
  })
})

describe('resolveMediaUrl', () => {
  it('prefixes relative media URLs with the API base', () => {
    const url = resolveMediaUrl('/media/dell_catalog/page_3.jpg')
    expect(url).toContain('/media/dell_catalog/page_3.jpg')
    expect(url).toBe(`${API_BASE}/media/dell_catalog/page_3.jpg`)
  })

  it('leaves absolute URLs untouched', () => {
    expect(resolveMediaUrl('https://example.com/x.jpg')).toBe(
      'https://example.com/x.jpg',
    )
  })

  it('returns null for null input', () => {
    expect(resolveMediaUrl(null)).toBeNull()
  })
})
