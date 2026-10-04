import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MessageList } from './MessageList'
import type { ChatMessage } from '../../hooks/useChat'

describe('MessageList', () => {
  it('shows the empty state when there are no messages', () => {
    render(
      <MessageList
        messages={[]}
        isLoading={false}
        onRetry={() => {}}
        onSuggestion={() => {}}
      />,
    )

    expect(screen.getByText('Ask your catalog anything')).toBeInTheDocument()
  })

  it('renders user and assistant messages', () => {
    const messages: ChatMessage[] = [
      { id: '1', role: 'user', content: 'What Dell products are shown?' },
      {
        id: '2',
        role: 'assistant',
        content: 'The catalog contains Dell laptops.',
        sources: [
          {
            chunk_id: 'text-001',
            page: 4,
            content_type: 'text',
            source: 'dell_catalog.pdf',
            image_path: null,
            image_url: null,
            snippet: 'Dell Precision laptop with Intel Core i7.',
          },
        ],
      },
    ]

    render(
      <MessageList
        messages={messages}
        isLoading={false}
        onRetry={() => {}}
        onSuggestion={() => {}}
      />,
    )

    expect(screen.getByText('What Dell products are shown?')).toBeInTheDocument()
    expect(
      screen.getByText('The catalog contains Dell laptops.'),
    ).toBeInTheDocument()
    expect(screen.getByText(/Sources \(1\)/)).toBeInTheDocument()
  })

  it('sends a suggestion when a suggestion chip is clicked', async () => {
    const user = userEvent.setup()
    const onSuggestion = vi.fn()

    render(
      <MessageList
        messages={[]}
        isLoading={false}
        onRetry={() => {}}
        onSuggestion={onSuggestion}
      />,
    )

    await user.click(
      screen.getByRole('button', {
        name: 'What is the Dell OptiPlex 3020?',
      }),
    )

    // The query must be submitted, not written into the DOM outside
    // React state (which previously left Send disabled).
    expect(onSuggestion).toHaveBeenCalledWith(
      'What is the Dell OptiPlex 3020?',
    )
  })

  it('shows a loading indicator while awaiting a response', () => {
    render(
      <MessageList
        messages={[]}
        isLoading={true}
        onRetry={() => {}}
        onSuggestion={() => {}}
      />,
    )

    expect(document.querySelector('.animate-bounce')).not.toBeNull()
  })

  it('shows an error state with a retry button on the last message', () => {
    const messages: ChatMessage[] = [
      { id: '1', role: 'user', content: 'hello' },
      {
        id: '2',
        role: 'assistant',
        content: '',
        error: 'Generation service is temporarily unavailable.',
      },
    ]

    render(
      <MessageList
        messages={messages}
        isLoading={false}
        onRetry={() => {}}
        onSuggestion={() => {}}
      />,
    )

    expect(
      screen.getByText(/Generation service is temporarily unavailable/),
    ).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
  })

  it('renders an image attached to a user message', () => {
    const messages: ChatMessage[] = [
      {
        id: '1',
        role: 'user',
        content: 'What is in this picture?',
        image: {
          filename: 'photo.png',
          media_type: 'image/png',
          data: 'aGVsbG8=',
          dataUrl: 'data:image/png;base64,aGVsbG8=',
        },
      },
      { id: '2', role: 'assistant', content: 'A red square.' },
    ]

    render(
      <MessageList
        messages={messages}
        isLoading={false}
        onRetry={() => {}}
        onSuggestion={() => {}}
      />,
    )

    // The user's own picture stays visible in the transcript, so an
    // answer about it can always be tied back to what was asked.
    const image = screen.getByAltText(/photo\.png/)
    expect(image).toHaveAttribute('src', 'data:image/png;base64,aGVsbG8=')
    expect(screen.getByText('What is in this picture?')).toBeInTheDocument()
  })
})
