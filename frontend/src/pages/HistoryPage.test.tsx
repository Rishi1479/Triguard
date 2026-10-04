import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as client from '../api/client'
import type { HistoryDetail, HistoryItem } from '../api/types'
import { scamResult } from '../test/fixtures'
import HistoryPage from './HistoryPage'

const item: HistoryItem = {
  id: 7, created_at: '2026-10-04T10:00:00+00:00', inputs_present: 'text+url', label: 'Likely Scam', risk_score: 87.4,
  url_domain: 'tcs-hiring.online', text_preview: 'Pay [PHONE] fee', reasons: ['High risk – Text: Asks you to pay a fee.'], latency_ms: 420,
}

const detail: HistoryDetail = {
  id: 7, created_at: item.created_at, inputs_present: 'text+url', submitted_text: 'Pay Rs 2000 fee, call [PHONE]',
  url_domain: 'tcs-hiring.online', image_submitted: true, latency_ms: 420, detail_available: true,
  feedback: [{ correct: false, comment: 'it was genuine', created_at: item.created_at }],
  result: scamResult,
}

beforeEach(() => vi.restoreAllMocks())

describe('HistoryPage', () => {
  it('lists checks and opens a full report when one is clicked', async () => {
    vi.spyOn(client, 'getHistory').mockResolvedValue([item])
    const spy = vi.spyOn(client, 'getHistoryDetail').mockResolvedValue(detail)
    render(<HistoryPage />)
    await userEvent.click(await screen.findByRole('button', { name: /Open full report for check 7/ }))
    expect(spy).toHaveBeenCalledWith(7, expect.anything())
    await waitFor(() => expect(screen.getByText('What was submitted')).toBeInTheDocument())
    expect(screen.getByText('Pay Rs 2000 fee, call [PHONE]')).toBeInTheDocument()
    expect(screen.getByText(/never saved/i)).toBeInTheDocument()
    expect(screen.getByRole('meter', { name: /scam risk/i })).toHaveAttribute('aria-valuenow', '87')
    expect(screen.getByText('Breakdown by input')).toBeInTheDocument()
    expect(screen.getByText(/Marked/)).toHaveTextContent('incorrect')
    expect(screen.getAllByRole('button', { name: 'What does this mean?' }).length).toBe(scamResult.reasons.length)
  })

  it('goes back to the list and notes legacy rows without details', async () => {
    vi.spyOn(client, 'getHistory').mockResolvedValue([item])
    vi.spyOn(client, 'getHistoryDetail').mockResolvedValue({ ...detail, detail_available: false, result: { ...scamResult, modality_results: [] }, feedback: [] })
    render(<HistoryPage />)
    await userEvent.click(await screen.findByRole('button', { name: /Open full report/ }))
    expect(await screen.findByText(/saved before per-input details/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /Back to recent checks/ }))
    expect(await screen.findByRole('heading', { name: 'Recent checks' })).toBeInTheDocument()
  })

  it('shows an error when a detail cannot be loaded', async () => {
    vi.spyOn(client, 'getHistory').mockResolvedValue([item])
    vi.spyOn(client, 'getHistoryDetail').mockRejectedValue(new client.ApiError('analysis_not_found', 'No such analysis.', 404))
    render(<HistoryPage />)
    await userEvent.click(await screen.findByRole('button', { name: /Open full report/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('No such analysis.')
  })
})
