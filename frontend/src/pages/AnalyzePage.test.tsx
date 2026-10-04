import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as client from '../api/client'
import { scamResult } from '../test/fixtures'
import AnalyzePage from './AnalyzePage'

beforeEach(() => vi.restoreAllMocks())
const analyzeBtn = () => screen.getByRole('button', { name: 'Analyze' })

describe('AnalyzePage', () => {
  it('disables Analyze until an input exists', async () => {
    render(<AnalyzePage />)
    expect(analyzeBtn()).toBeDisabled()
    await userEvent.type(screen.getByLabelText(/Job or internship text/), 'hello')
    expect(analyzeBtn()).toBeEnabled()
    await userEvent.clear(screen.getByLabelText(/Job or internship text/))
    expect(analyzeBtn()).toBeDisabled()
  })

  it('enables with only a URL and flags an invalid URL', async () => {
    render(<AnalyzePage />)
    const url = screen.getByLabelText(/Application link/)
    await userEvent.type(url, 'not a url')
    expect(screen.getByRole('alert')).toHaveTextContent(/spaces|valid/)
    expect(analyzeBtn()).toBeDisabled()
    await userEvent.clear(url)
    await userEvent.type(url, 'tcs.com/careers')
    expect(analyzeBtn()).toBeEnabled()
  })

  it('validates image type and shows a preview for a valid image', async () => {
    const urlApi = URL as unknown as { createObjectURL?: unknown; revokeObjectURL?: unknown }
    urlApi.createObjectURL = vi.fn(() => 'blob:preview')
    urlApi.revokeObjectURL = vi.fn()
    render(<AnalyzePage />)
    const input = screen.getByLabelText('Choose an image file')
    await userEvent.upload(input, new File(['x'], 'doc.pdf', { type: 'application/pdf' }), { applyAccept: false })
    expect(screen.getByRole('alert')).toHaveTextContent(/PNG, JPG or WebP/)
    expect(analyzeBtn()).toBeDisabled()
    await userEvent.upload(input, new File(['x'], 'letter.png', { type: 'image/png' }))
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.getByAltText('Preview of letter.png')).toBeInTheDocument()
    expect(analyzeBtn()).toBeEnabled()
  })

  it('submits, shows progress, then the result', async () => {
    let resolve!: (v: typeof scamResult) => void
    const spy = vi.spyOn(client, 'analyze').mockReturnValue(new Promise((r) => { resolve = r }))
    render(<AnalyzePage />)
    await userEvent.type(screen.getByLabelText(/Job or internship text/), 'Pay fee')
    await userEvent.type(screen.getByLabelText(/Application link/), 'http://tcs-hiring.online')
    await userEvent.click(analyzeBtn())
    expect(screen.getByRole('status')).toHaveTextContent(/Reading the text/)
    expect(spy.mock.calls[0][0]).toMatchObject({ text: 'Pay fee', url: 'http://tcs-hiring.online' })
    resolve(scamResult)
    await waitFor(() => expect(screen.getByText('Likely Scam')).toBeInTheDocument())
  })

  it('shows server errors and lets the user retry', async () => {
    vi.spyOn(client, 'analyze').mockRejectedValue(new client.ApiError('rate_limited', 'Too many requests.', 429))
    render(<AnalyzePage />)
    await userEvent.type(screen.getByLabelText(/Job or internship text/), 'hello')
    await userEvent.click(analyzeBtn())
    expect(await screen.findByRole('alert')).toHaveTextContent('Too many requests.')
    expect(analyzeBtn()).toBeEnabled()
  })
})

describe('AnalyzePage — example chips and input progress', () => {
  it('offers the demo examples', () => {
    render(<AnalyzePage />)
    expect(screen.getByRole('button', { name: /Genuine offer letter/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Fake offer letter/ })).toBeInTheDocument()
  })

  it('fills the form from an example, including its link', async () => {
    render(<AnalyzePage />)
    await userEvent.click(screen.getByRole('button', { name: /Genuine job posting/ }))
    const textarea = screen.getByRole('textbox', { name: /job or internship text/i }) as HTMLTextAreaElement
    expect(textarea.value).toContain('Wealth Bank is hiring a Relationship Manager')
    expect(screen.getByRole('textbox', { name: /application link/i })).toHaveValue(
      'https://careers.wealthbank.com/apply',
    )
  })

  it('enables Analyze once an example is loaded', async () => {
    render(<AnalyzePage />)
    expect(screen.getByRole('button', { name: 'Analyze' })).toBeDisabled()
    await userEvent.click(screen.getByRole('button', { name: /Fake offer letter/ }))
    expect(screen.getByRole('button', { name: 'Analyze' })).toBeEnabled()
  })

  it('tracks which inputs are provided', async () => {
    render(<AnalyzePage />)
    expect(screen.getByText('Add at least one.')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /Genuine job posting/ }))
    expect(screen.getByText(/More inputs let us cross-check/)).toBeInTheDocument()
  })

  it('clears the form', async () => {
    render(<AnalyzePage />)
    await userEvent.click(screen.getByRole('button', { name: /Fake offer letter/ }))
    await userEvent.click(screen.getByRole('button', { name: 'Clear' }))
    expect(screen.getByRole('textbox', { name: /job or internship text/i })).toHaveValue('')
    expect(screen.getByRole('button', { name: 'Analyze' })).toBeDisabled()
  })
})
