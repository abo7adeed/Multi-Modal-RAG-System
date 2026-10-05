import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { parseSSEFrames, queryRAGStream, RAGApiError } from './ragApi'

/**
 * Build a Response whose body delivers the given chunks, so tests can
 * control exactly where the network splits the stream.
 */
function streamResponse(chunks: string[], status = 200): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk))
      }
      controller.close()
    },
  })

  return new Response(body, {
    status,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
}

describe('parseSSEFrames', () => {
  it('decodes a sources frame', () => {
    const body = frame('sources', { sources: [{ chunk_id: 'text-001' }] })

    const { frames, rest } = parseSSEFrames(body)

    expect(rest).toBe('')
    expect(frames).toEqual([
      { type: 'sources', sources: [{ chunk_id: 'text-001' }] },
    ])
  })

  it('decodes token, done and error frames', () => {
    const body =
      frame('token', { text: 'Dell ' }) +
      frame('done', {}) +
      frame('error', { message: 'Boom', type: 'ProviderTimeoutError' })

    const { frames } = parseSSEFrames(body)

    expect(frames[0]).toEqual({ type: 'token', text: 'Dell ' })
    expect(frames[1]).toEqual({ type: 'done' })
    expect(frames[2]).toEqual({
      type: 'error',
      message: 'Boom',
      errorType: 'ProviderTimeoutError',
    })
  })

  it('keeps an incomplete trailing frame as the remainder', () => {
    // Chunk boundaries from the network fall anywhere, including
    // mid-frame. Dropping this would lose the last token.
    const { frames, rest } = parseSSEFrames(
      frame('token', { text: 'Dell ' }) + 'event: token\ndata: {"text":"Prec',
    )

    expect(frames).toHaveLength(1)
    expect(rest).toBe('event: token\ndata: {"text":"Prec')
  })

  it('reassembles a frame split across two chunks', () => {
    const complete = frame('token', { text: 'Dell' })
    const splitAt = 15

    const first = parseSSEFrames(complete.slice(0, splitAt))
    expect(first.frames).toHaveLength(0)

    const second = parseSSEFrames(first.rest + complete.slice(splitAt))
    expect(second.frames).toEqual([{ type: 'token', text: 'Dell' }])
  })

  it('preserves newlines inside a token', () => {
    // The payload is JSON-encoded, so a newline cannot break framing.
    const { frames } = parseSSEFrames(frame('token', { text: 'one\ntwo' }))

    expect(frames[0]).toEqual({ type: 'token', text: 'one\ntwo' })
  })

  it('ignores unknown events so the server can add them safely', () => {
    const { frames } = parseSSEFrames(
      frame('token', { text: 'Dell' }) + frame('progress', { pct: 50 }),
    )

    expect(frames).toHaveLength(1)
    expect(frames[0].type).toBe('token')
  })

  it('skips a malformed frame rather than throwing', () => {
    const { frames } = parseSSEFrames(
      'event: token\ndata: {not json}\n\n' + frame('token', { text: 'Dell' }),
    )

    // One bad frame must not lose the rest of a working answer.
    expect(frames).toEqual([{ type: 'token', text: 'Dell' }])
  })
})

describe('queryRAGStream', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('reports sources, tokens and completion', async () => {
    vi.mocked(fetch).mockResolvedValue(
      streamResponse([
        frame('sources', { sources: [{ chunk_id: 'text-001' }] }),
        frame('token', { text: 'Dell ' }),
        frame('token', { text: 'laptops.' }),
        frame('done', {}),
      ]),
    )

    const sources: unknown[] = []
    const tokens: string[] = []
    let done = false

    await queryRAGStream('What Dell products are shown?', null, {
      onSources: (value) => sources.push(value),
      onToken: (text) => tokens.push(text),
      onDone: () => {
        done = true
      },
    })

    expect(sources).toEqual([[{ chunk_id: 'text-001' }]])
    expect(tokens.join('')).toBe('Dell laptops.')
    expect(done).toBe(true)
  })

  it('delivers tokens progressively rather than all at the end', async () => {
    // A single network chunk for the whole answer would still pass a
    // "final text" assertion, so each token gets its own chunk and the
    // ordering of handler calls is what is checked.
    vi.mocked(fetch).mockResolvedValue(
      streamResponse([
        frame('token', { text: 'Dell ' }),
        frame('token', { text: 'Precision ' }),
        frame('token', { text: 'laptop.' }),
        frame('done', {}),
      ]),
    )

    const order: string[] = []

    await queryRAGStream('What is shown?', null, {
      onToken: (text) => order.push(`token:${text}`),
      onDone: () => order.push('done'),
    })

    expect(order).toEqual([
      'token:Dell ',
      'token:Precision ',
      'token:laptop.',
      'done',
    ])
  })

  it('posts to the stream endpoint with SSE accept header', async () => {
    vi.mocked(fetch).mockResolvedValue(
      streamResponse([frame('token', { text: 'ok' }), frame('done', {})]),
    )

    await queryRAGStream('q', null, {})

    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/rag/stream'),
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ Accept: 'text/event-stream' }),
      }),
    )
  })

  it('throws the API error when the stream reports one', async () => {
    vi.mocked(fetch).mockResolvedValue(
      streamResponse([
        frame('token', { text: 'Dell ' }),
        frame('error', {
          message: 'Generation took too long to respond.',
          type: 'ProviderTimeoutError',
        }),
      ]),
    )

    await expect(queryRAGStream('q', null, {})).rejects.toThrow(
      'Generation took too long to respond.',
    )
  })

  it('maps a streamed error to the timeout type', async () => {
    vi.mocked(fetch).mockResolvedValue(
      streamResponse([
        frame('error', { message: 'Slow.', type: 'ProviderTimeoutError' }),
      ]),
    )

    await expect(queryRAGStream('q', null, {})).rejects.toMatchObject({
      errorType: 'timeout',
    })
  })

  it('fails when the stream ends without a done event', async () => {
    // A truncated stream must not look like a finished answer.
    vi.mocked(fetch).mockResolvedValue(
      streamResponse([frame('token', { text: 'Dell ' })]),
    )

    await expect(queryRAGStream('q', null, {})).rejects.toThrow(
      /cut off/i,
    )
  })

  it('uses the real HTTP status for a pre-stream failure', async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(
        JSON.stringify({
          error: {
            message: 'Query must not be empty.',
            type: 'InvalidRequestError',
          },
        }),
        { status: 400 },
      ),
    )

    await expect(queryRAGStream('   ', null, {})).rejects.toMatchObject({
      status: 400,
      errorType: 'invalid_request',
    })
  })

  it('reports a missing body rather than hanging', async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(null, { status: 200 }),
    )

    await expect(queryRAGStream('q', null, {})).rejects.toThrow(
      /not supported/i,
    )
  })

  it('surfaces a network failure as a RAGApiError', async () => {
    vi.mocked(fetch).mockRejectedValue(new TypeError('Failed to fetch'))

    const error = await queryRAGStream('q', null, {}).catch((err) => err)

    expect(error).toBeInstanceOf(RAGApiError)
    expect(error.errorType).toBe('network')
  })

  it('forwards an attachment and max_sources', async () => {
    vi.mocked(fetch).mockResolvedValue(
      streamResponse([frame('done', {})]),
    )

    await queryRAGStream(
      'What is this?',
      { filename: 'a.jpg', media_type: 'image/jpeg', data: 'YmFzZTY0' },
      {},
      undefined,
      5,
    )

    const body = JSON.parse(
      vi.mocked(fetch).mock.calls[0][1]?.body as string,
    )
    expect(body.image.filename).toBe('a.jpg')
    expect(body.max_sources).toBe(5)
  })
})