import { describe, it, expect, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ChatInput } from './ChatInput'
import { MAX_ATTACHMENT_BYTES } from '../../services/ragApi'

function renderInput(onSend = vi.fn(), isLoading = false) {
  render(<ChatInput onSend={onSend} isLoading={isLoading} />)
  return { onSend, user: userEvent.setup() }
}

function imageFile(
  name = 'photo.png',
  type = 'image/png',
  bytes = 'hello',
): File {
  return new File([bytes], name, { type })
}

// The picker is visually hidden, so it is addressed by its label
// rather than by visibility.
function fileInput(): HTMLInputElement {
  const input = document.querySelector(
    'input[type="file"]',
  ) as HTMLInputElement
  expect(input).not.toBeNull()
  return input
}

describe('ChatInput attachments', () => {
  it('previews a picked image and sends it with the question', async () => {
    const { onSend, user } = renderInput()

    await user.upload(fileInput(), imageFile())

    const preview = await screen.findByAltText(/photo\.png/)
    expect(preview).toHaveAttribute(
      'src',
      `data:image/png;base64,${btoa('hello')}`,
    )

    await user.type(
      screen.getByLabelText('Ask about your documents'),
      'What is this?',
    )
    await user.click(screen.getByRole('button', { name: 'Send message' }))

    expect(onSend).toHaveBeenCalledWith(
      'What is this?',
      expect.objectContaining({
        filename: 'photo.png',
        media_type: 'image/png',
        data: btoa('hello'),
      }),
    )
  })

  it('clears the attachment after sending', async () => {
    const { user } = renderInput()

    await user.upload(fileInput(), imageFile())
    await screen.findByAltText(/photo\.png/)

    await user.type(
      screen.getByLabelText('Ask about your documents'),
      'What is this?',
    )
    await user.keyboard('{Enter}')

    await waitFor(() => {
      expect(screen.queryByAltText(/photo\.png/)).not.toBeInTheDocument()
    })
    expect(
      (screen.getByLabelText('Ask about your documents') as HTMLTextAreaElement)
        .value,
    ).toBe('')
  })

  it('removes an attached image on request', async () => {
    const { user } = renderInput()

    await user.upload(fileInput(), imageFile())
    await screen.findByAltText(/photo\.png/)

    await user.click(
      screen.getByRole('button', { name: 'Remove attached image' }),
    )

    expect(screen.queryByAltText(/photo\.png/)).not.toBeInTheDocument()
  })

  it('replaces the attachment when a second image is picked', async () => {
    const { user } = renderInput()

    await user.upload(fileInput(), imageFile('first.png'))
    await screen.findByAltText(/first\.png/)

    await user.upload(fileInput(), imageFile('second.png'))

    await screen.findByAltText(/second\.png/)
    expect(screen.queryByAltText(/first\.png/)).not.toBeInTheDocument()
  })

  it('rejects a file that is not an image', async () => {
    // accept= is only a picker hint, so a drag-and-drop or a renamed
    // file can still reach the handler: the type is re-checked there.
    const { onSend } = renderInput()

    fireEvent.change(fileInput(), {
      target: {
        files: [
          new File(['%PDF-1.4'], 'invoice.png', {
            type: 'application/pdf',
          }),
        ],
      },
    })

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /Attach a JPG, PNG, WEBP or GIF image/,
    )
    expect(
      screen.queryByAltText(/invoice\.png/),
    ).not.toBeInTheDocument()
    expect(onSend).not.toHaveBeenCalled()
  })

  it('rejects an oversized image before reading it', async () => {
    const { user } = renderInput()

    const tooBig = new File(['x'], 'huge.png', { type: 'image/png' })
    Object.defineProperty(tooBig, 'size', {
      value: MAX_ATTACHMENT_BYTES + 1,
    })

    await user.upload(fileInput(), tooBig)

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /The limit is 5 MB/,
    )
    expect(screen.queryByAltText(/huge\.png/)).not.toBeInTheDocument()
  })

  it('keeps Send disabled until a question is typed', async () => {
    const { user } = renderInput()

    await user.upload(fileInput(), imageFile())
    await screen.findByAltText(/photo\.png/)

    // An image alone is not a question: the API requires text, and
    // sending an empty one would fail.
    expect(
      screen.getByRole('button', { name: 'Send message' }),
    ).toBeDisabled()

    await user.type(
      screen.getByLabelText('Ask about your documents'),
      'Describe it',
    )

    expect(
      screen.getByRole('button', { name: 'Send message' }),
    ).toBeEnabled()
  })
})
