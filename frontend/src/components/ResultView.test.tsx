import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as client from '../api/client'
import { safeResult, scamResult } from '../test/fixtures'
import ResultView from './ResultView'

beforeEach(() => vi.restoreAllMocks())

describe('ResultView', () => {
  it('shows the verdict, gauge value and summary', () => {
    render(<ResultView result={scamResult} onReset={() => {}} />)
    expect(screen.getByText('Likely Scam')).toBeInTheDocument()
    const gauge = screen.getByRole('meter', { name: /scam risk/i })
    expect(gauge).toHaveAttribute('aria-valuenow', '87')
    expect(gauge).toHaveAttribute('aria-valuetext', expect.stringContaining('Likely Scam'))
    expect(screen.getByText(/do not pay any money/i)).toBeInTheDocument()
  })

  it('lists reasons with highlighted evidence and accessible severity', () => {
    render(<ResultView result={scamResult} onReset={() => {}} />)
    const marks = document.querySelectorAll('mark')
    expect([...marks].map((m) => m.textContent)).toContain('“Pay Rs 2000 registration fee”')
    expect(screen.getAllByText(/High risk\./).length).toBeGreaterThanOrEqual(2) // sr-only severity text
    expect(screen.getByRole('heading', { name: 'Why?' })).toBeInTheDocument()
  })

  it('shows prominent cross-modal warnings only when present', () => {
    const { rerender } = render(<ResultView result={scamResult} onReset={() => {}} />)
    expect(screen.getByRole('alert')).toHaveTextContent(/don’t match each other/)
    rerender(<ResultView result={safeResult} onReset={() => {}} />)
    expect(screen.queryByText(/don’t match each other/)).not.toBeInTheDocument()
    expect(screen.getByText('Safe')).toBeInTheDocument()
  })

  it('renders per-modality cards including not-provided state', () => {
    render(<ResultView result={scamResult} onReset={() => {}} />)
    expect(screen.getByText('Not provided')).toBeInTheDocument() // image
    expect(screen.getByText('tcs-hiring.online')).toBeInTheDocument()
    expect(screen.getByText('93%')).toBeInTheDocument()
    expect(screen.getByText('fee', { selector: 'span.bg-yellow-100' })).toBeInTheDocument()
  })

  it('includes safety tips with reporting links and the disclaimer', () => {
    render(<ResultView result={scamResult} onReset={() => {}} />)
    expect(screen.getByRole('link', { name: 'cybercrime.gov.in' })).toHaveAttribute('href', 'https://cybercrime.gov.in')
    expect(screen.getByText('1930')).toBeInTheDocument()
    expect(screen.getByText(/Never pay/)).toBeInTheDocument()
    expect(screen.getByText(/not legal advice/i)).toBeInTheDocument()
  })

  it('sends feedback and confirms', async () => {
    const spy = vi.spyOn(client, 'sendFeedback').mockResolvedValue({ id: 1, status: 'recorded' })
    render(<ResultView result={scamResult} onReset={() => {}} />)
    await userEvent.click(screen.getByRole('button', { name: 'Yes' }))
    expect(spy).toHaveBeenCalledWith(7, true)
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(/Thanks/))
  })

  it('shows an error when feedback fails and calls onReset', async () => {
    vi.spyOn(client, 'sendFeedback').mockRejectedValue(new client.ApiError('network', 'x'))
    const reset = vi.fn()
    render(<ResultView result={scamResult} onReset={reset} />)
    await userEvent.click(screen.getByRole('button', { name: 'No' }))
    await waitFor(() => expect(screen.getByRole('alert', { name: '' })).toBeTruthy())
    await userEvent.click(screen.getByRole('button', { name: /check another/i }))
    expect(reset).toHaveBeenCalled()
  })
})
