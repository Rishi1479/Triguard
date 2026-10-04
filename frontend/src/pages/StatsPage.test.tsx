import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as client from '../api/client'
import type { Stats } from '../api/types'
import StatsPage from './StatsPage'

const STATS: Stats = {
  total_analyses: 8,
  label_counts: { Safe: 3, Suspicious: 1, 'Likely Scam': 4 },
  input_combinations: { 'text+image+url': 5, text: 3 },
  avg_risk_score: 54.2,
  avg_latency_ms: 412,
  feedback: { total: 4, marked_correct: 3, accuracy: 0.75 },
}

beforeEach(() => vi.restoreAllMocks())

describe('StatsPage', () => {
  it('shows the key figures once loaded', async () => {
    vi.spyOn(client, 'getStats').mockResolvedValue(STATS)
    render(<StatsPage />)
    await waitFor(() => expect(screen.getByText('Checks run')).toBeInTheDocument())
    expect(screen.getByText('8')).toBeInTheDocument()
    expect(screen.getByText('54.2')).toBeInTheDocument()
    expect(screen.getByText('412')).toBeInTheDocument()
    expect(screen.getByText('Average response')).toBeInTheDocument()
    expect(screen.getByText('75%')).toBeInTheDocument()
    expect(screen.getByText('3 of 4 ratings')).toBeInTheDocument()
  })

  it('charts the verdict distribution and offers a table view', async () => {
    vi.spyOn(client, 'getStats').mockResolvedValue(STATS)
    render(<StatsPage />)
    await waitFor(() => expect(screen.getByText('4 · 50%')).toBeInTheDocument())
    await userEvent.click(screen.getByRole('button', { name: 'Show table' }))
    expect(screen.getByRole('table', { name: 'Verdict counts' })).toBeInTheDocument()
  })

  it('gives input combinations readable names', async () => {
    vi.spyOn(client, 'getStats').mockResolvedValue(STATS)
    render(<StatsPage />)
    await waitFor(() => expect(screen.getByText('All three')).toBeInTheDocument())
    expect(screen.getByText('Text only')).toBeInTheDocument()
  })

  it('states that these are usage counts, not model accuracy', async () => {
    vi.spyOn(client, 'getStats').mockResolvedValue(STATS)
    render(<StatsPage />)
    await waitFor(() => expect(screen.getByText(/not model accuracy/i)).toBeInTheDocument())
  })

  it('shows an empty state before anything has been analysed', async () => {
    vi.spyOn(client, 'getStats').mockResolvedValue({
      ...STATS,
      total_analyses: 0,
      label_counts: {},
      input_combinations: {},
      avg_risk_score: null,
      avg_latency_ms: null,
      feedback: { total: 0, marked_correct: 0, accuracy: null },
    })
    render(<StatsPage />)
    await waitFor(() => expect(screen.getByText(/Nothing analysed yet/i)).toBeInTheDocument())
  })

  it('reports a load failure', async () => {
    vi.spyOn(client, 'getStats').mockRejectedValue(new client.ApiError('network', 'Server unreachable'))
    render(<StatsPage />)
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/Server unreachable/))
  })

  it('stays silent when the request was cancelled', async () => {
    vi.spyOn(client, 'getStats').mockRejectedValue(new client.ApiError('cancelled', 'x'))
    render(<StatsPage />)
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument())
  })
})
