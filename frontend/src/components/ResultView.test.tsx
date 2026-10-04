import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as client from '../api/client'
import { safeResult, scamResult } from '../test/fixtures'
import ResultView from './ResultView'

beforeEach(() => vi.restoreAllMocks())

describe('ResultView – provisional (abstained) verdicts', () => {
  it('shows the provisional banner and marker when the backend abstained', () => {
    render(<ResultView result={{ ...scamResult, abstained: true, evidence_strength: 'low' }} onReset={() => {}} />)
    expect(screen.getByRole('status')).toHaveTextContent(/not enough evidence/i)
    expect(screen.getByText('Provisional')).toBeInTheDocument()
    expect(screen.getByText(/add more to go on/i)).toBeInTheDocument()
  })

  it('still renders the label and gauge so existing clients keep working', () => {
    render(<ResultView result={{ ...scamResult, abstained: true, evidence_strength: 'low' }} onReset={() => {}} />)
    expect(screen.getByText('Likely Scam')).toBeInTheDocument()
    expect(screen.getByRole('meter', { name: /scam risk/i })).toHaveAttribute('aria-valuenow', '87')
  })

  it('shows nothing extra for a normal verdict', () => {
    render(<ResultView result={safeResult} onReset={() => {}} />)
    expect(screen.queryByText('Provisional')).not.toBeInTheDocument()
    expect(screen.queryByText(/not enough evidence/i)).not.toBeInTheDocument()
  })

  it('treats a missing abstained field as a normal verdict', () => {
    render(<ResultView result={scamResult} onReset={() => {}} />)
    expect(screen.queryByText('Provisional')).not.toBeInTheDocument()
  })
})

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

describe('reason explanations', () => {
  it('each reason can be expanded into a plain-English explanation', async () => {
    render(<ResultView result={scamResult} onReset={() => {}} />)
    const buttons = screen.getAllByRole('button', { name: 'What does this mean?' })
    expect(buttons).toHaveLength(scamResult.reasons.length)
    await userEvent.click(buttons[1]) // the payment-request reason
    expect(screen.getByText('You are asked to pay money')).toBeInTheDocument()
    expect(screen.getByText(/do not charge candidates/i)).toBeInTheDocument()
    expect(screen.getAllByText(/cybercrime.gov.in/).length).toBeGreaterThan(1) // in the explanation and the safety tips
    expect(buttons[1]).toHaveAttribute('aria-expanded', 'true')
    await userEvent.click(screen.getByRole('button', { name: 'Hide explanation' }))
    expect(screen.queryByText('You are asked to pay money')).not.toBeInTheDocument()
  })

  it('explains model-driven notes without pretending they are proof', async () => {
    const result = { ...scamResult, reasons: ['Note – Text: the AI model’s score was driven mostly by the words “fee”, “pay”. Single words are only hints, not proof.'] }
    render(<ResultView result={result} onReset={() => {}} />)
    await userEvent.click(screen.getByRole('button', { name: 'What does this mean?' }))
    expect(screen.getByText('Words that influenced the AI score')).toBeInTheDocument()
    expect(screen.getByText(/never proves a scam/i)).toBeInTheDocument()
  })

  it('falls back to a generic explanation for unknown reasons', async () => {
    const result = { ...scamResult, reasons: ['High risk – Text: Some brand new rule fired.'], cross_modal_flags: [], modality_results: [] }
    render(<ResultView result={result} onReset={() => {}} />)
    await userEvent.click(screen.getByRole('button', { name: 'What does this mean?' }))
    expect(screen.getByText('Warning sign')).toBeInTheDocument()
  })
})
