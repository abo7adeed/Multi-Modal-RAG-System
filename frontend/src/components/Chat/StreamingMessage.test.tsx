import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { MessageList } from './MessageList'
import { useChat } from '../../hooks/useChat'

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
}

/**
 * A Response whose body stays open, so a test can push tokens one at
 * a time and observe the UI between them.
 */
function controllableStream() {
  const encoder = new TextEncoder()
  let controller: ReadableStreamDefaultController<Uint8Array> | null = null

  const body = new ReadableStream<Uint8Array>({
    start(streamController) {
      controller = streamController
    },
  })

  return {
    response: new Response(body, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    }),
    push(text: string) {
      controller?.enqueue(encoder.encode(text))
    },
    close() {
      controller?.close()
    },
  }
}

/**
 * The list plus the hook, so the test drives the same path the app
 * uses rather than a hand-built props object.
 */
function TestChat() {
  const { messages, isLoading, ask } = useChat()
  const [text, setText] = useState('')

  return (
    <div>
      <input
        aria-label="question"
        value={text}
        onChange={(event) => setText(event.target.value)}
      />
      <button type="button" onClick={() => void ask(text, null)}>
        send
      </button>
      <MessageList
        messages={messages}
        isLoading={isLoading}
        onRetry={() => {}}
        onSuggestion={() => {}}
      />
    </div>
  )
}

describe('streaming answer rendering', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders tokens progressively as they arrive', async () => {
    const stream = controllableStream()
    vi.mocked(fetch).mockResolvedValue(stream.response)

    render(<TestChat />)

    const input = screen.getByLabelText('question')
    await input_event(input, 'What Dell products are shown?')

    // Before any token: only the typing indicator.
    await waitFor(() => {
      expect(screen.getByText('What Dell products are shown?')).toBeInTheDocument()
    })

    stream.push(frame('token', { text: 'Dell ' }))
    await waitFor(() => {
      expect(screen.getByText(/Dell/)).toBeInTheDocument()
    })

    // The partial answer is visible before the rest has arrived.
    expect(screen.getByText('Dell')).toBeInTheDocument()
    expect(screen.queryByText(/Precision laptop/)).not.toBeInTheDocument()

    stream.push(frame('token', { text: 'Precision laptop with ' }))
    await waitFor(() => {
      expect(screen.getByText('Dell Precision laptop with')).toBeInTheDocument()
    })

    stream.push(frame('token', { text: 'Intel Core i7.' }))
    stream.push(frame('done', {}))
    stream.close()

    await waitFor(() => {
      expect(
        screen.getByText('Dell Precision laptop with Intel Core i7.'),
      ).toBeInTheDocument()
    })
  })

  it('shows citations before the answer has finished', async () => {
    // The server sends sources first precisely so the UI can render
    // them while the text is still being written.
    const stream = controllableStream()
    vi.mocked(fetch).mockResolvedValue(stream.response)

    render(<TestChat />)

    await input_event(screen.getByLabelText('question'), 'What is on page 4?')

    stream.push(
      frame('sources', {
        sources: [
          {
            chunk_id: 'text-001',
            page: 4,
            content_type: 'text',
            source: 'dell_catalog.pdf',
            image_path: null,
            image_url: null,
            snippet: 'Dell Precision laptop with Intel Core i7.',
            score: 0.032,
            kind: 'document',
          },
        ],
      }),
    )
    stream.push(frame('token', { text: 'A Dell Precision laptop.' }))

    await waitFor(() => {
      expect(screen.getByText(/A Dell Precision laptop/)).toBeInTheDocument()
    })

    // Page 4 is visible while generation is still in progress.
    expect(screen.getByText(/Page 4/)).toBeInTheDocument()

    stream.push(frame('done', {}))
    stream.close()

    await waitFor(() => {
      expect(screen.getByText(/A Dell Precision laptop/)).toBeInTheDocument()
    })
  })

  it('keeps partial text and flags it when the stream fails', async () => {
    const stream = controllableStream()
    vi.mocked(fetch).mockResolvedValue(stream.response)

    render(<TestChat />)

    await input_event(screen.getByLabelText('question'), 'What is page 4?')

    stream.push(frame('token', { text: 'A Dell Precision' }))
    await waitFor(() => {
      expect(screen.getByText(/A Dell Precision/)).toBeInTheDocument()
    })

    stream.push(
      frame('error', {
        message: 'Generation took too long to respond.',
        type: 'ProviderTimeoutError',
      }),
    )
    stream.close()

    // The real content is kept rather than thrown away, but it is
    // marked as possibly incomplete.
    await waitFor(() => {
      expect(screen.getByText(/incomplete/i)).toBeInTheDocument()
    })
    expect(screen.getByText('A Dell Precision')).toBeInTheDocument()
  })

  it('does not show a typing indicator once tokens have arrived', async () => {
    const stream = controllableStream()
    vi.mocked(fetch).mockResolvedValue(stream.response)

    const { container } = render(<TestChat />)

    await input_event(screen.getByLabelText('question'), 'What is page 4?')

    // The bouncing dots are shown only while waiting for the first
    // token; afterwards the caret in the bubble takes over.
    await waitFor(() => {
      expect(container.querySelectorAll('.animate-bounce').length).toBe(3)
    })

    stream.push(frame('token', { text: 'A Dell' }))
    await waitFor(() => {
      expect(container.querySelectorAll('.animate-bounce').length).toBe(0)
    })

    stream.push(frame('done', {}))
    stream.close()

    // Let the completion state settle so no state update escapes the
    // test body.
    await waitFor(() => {
      expect(screen.getByText('A Dell')).toBeInTheDocument()
    })
  })
})

/** Type into the controlled input and submit. */
async function input_event(
  element: HTMLElement,
  value: string,
): Promise<void> {
  const userEvent = (await import('@testing-library/user-event')).default
  await userEvent.clear(element)
  await userEvent.type(element, value)
  await userEvent.click(screen.getByRole('button', { name: 'send' }))
}
describe('follow-up history', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('sends the previous turns with a follow-up question', async () => {
    vi.mocked(fetch).mockResolvedValue(
      streamResponse([frame('token', { text: 'ok' }), frame('done', {})]),
    )

    render(<TestChat />)

    await input_event(screen.getByLabelText('question'), 'What is the Dell OptiPlex 3020?')
    await waitFor(() => {
      expect(vi.mocked(fetch).mock.calls.length).toBe(1)
    })
    expect(bodyOf(0).history).toBeUndefined()

    await input_event(screen.getByLabelText('question'), 'What about its warranty?')
    await waitFor(() => {
      expect(vi.mocked(fetch).mock.calls.length).toBe(2)
    })

    // The first question alone is what makes the follow-up
    // resolvable, so it has to travel with the request.
    expect(bodyOf(1).history).toEqual([
      { role: 'user', content: 'What is the Dell OptiPlex 3020?' },
      { role: 'assistant', content: 'ok' },
    ])
    expect(bodyOf(1).query).toBe('What about its warranty?')
  })

  it('never replays a failed answer as conversation history', async () => {
    vi.mocked(fetch).mockResolvedValue(
      streamResponse([
        frame('error', { message: 'Boom.', type: 'ProviderTimeoutError' }),
      ]),
    )

    render(<TestChat />)

    await input_event(screen.getByLabelText('question'), 'What is the OptiPlex 3020?')
    await waitFor(() => {
      expect(screen.getByText(/Boom/)).toBeInTheDocument()
    })

    vi.mocked(fetch).mockResolvedValue(
      streamResponse([frame('token', { text: 'ok' }), frame('done', {})]),
    )

    await input_event(screen.getByLabelText('question'), 'And the price?')
    await waitFor(() => {
      expect(vi.mocked(fetch).mock.calls.length).toBe(2)
    })

    // The user's question is legitimate context, but the failed
    // answer is not: replaying it would teach the model that a
    // question can be answered with an error.
    expect(bodyOf(1).history).toEqual([
      { role: 'user', content: 'What is the OptiPlex 3020?' },
    ])
  })
})

/** Parse the JSON body of the nth fetch call. */
function bodyOf(index: number): Record<string, unknown> {
  return JSON.parse(
    vi.mocked(fetch).mock.calls[index][1]?.body as string,
  )
}

/** A Response that delivers a complete SSE body at once. */
function streamResponse(chunks: string[]): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
  return new Response(body, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}
