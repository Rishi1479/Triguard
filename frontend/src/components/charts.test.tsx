import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { CountBars, DataTable, DivergingWords, LabelDistribution, ModalityBars, RiskBandMeter } from './charts'
import { MODALITY_ICON } from './Icons'

describe('RiskBandMeter', () => {
  it('names the band boundaries so the thresholds are legible', () => {
    render(<RiskBandMeter score={87} label="Likely Scam" />)
    expect(screen.getByText(/30 — Suspicious above/)).toBeInTheDocument()
    expect(screen.getByText(/70 — Likely scam above/)).toBeInTheDocument()
    expect(screen.getByText(/Risk 87% of 100/)).toBeInTheDocument()
  })

  it('clamps an out-of-range score rather than overflowing the track', () => {
    const { container } = render(<RiskBandMeter score={140} label="Likely Scam" />)
    const marker = container.querySelector('g[transform]')
    expect(marker?.getAttribute('transform')).toBe('translate(100 0)')
  })

  it('does not restate the verdict (the badge above it already does)', () => {
    render(<RiskBandMeter score={10} label="Safe" />)
    expect(screen.queryByText('Safe')).not.toBeInTheDocument()
  })
})

describe('ModalityBars', () => {
  const data = [
    { name: 'Text', value: 0.93, icon: MODALITY_ICON.text },
    { name: 'Image', value: null, icon: MODALITY_ICON.image },
    { name: 'Link', value: 0.85, icon: MODALITY_ICON.url },
  ]

  it('direct-labels every bar and marks the missing input', () => {
    render(<ModalityBars data={data} title="Scam probability per input" />)
    expect(screen.getByText('93%')).toBeInTheDocument()
    expect(screen.getByText('85%')).toBeInTheDocument()
    expect(screen.getByText('not given')).toBeInTheDocument()
  })

  it('exposes each row by accessible name, so value is never colour-only', () => {
    render(<ModalityBars data={data} title="t" />)
    expect(screen.getByLabelText('Text: 93% scam probability')).toBeInTheDocument()
    expect(screen.getByLabelText('Image: not provided')).toBeInTheDocument()
  })

  it('each row is focusable for keyboard users', () => {
    render(<ModalityBars data={data} title="t" />)
    expect(screen.getByLabelText('Text: 93% scam probability')).toHaveAttribute('tabIndex', '0')
  })
})

describe('DivergingWords', () => {
  const data = [
    { word: 'fee', weight: 0.4 },
    { word: 'team', weight: -0.2 },
  ]

  it('labels both poles in a legend', () => {
    render(<DivergingWords data={data} title="Word influence" />)
    expect(screen.getByText('pushed toward scam')).toBeInTheDocument()
    expect(screen.getByText('pushed toward genuine')).toBeInTheDocument()
  })

  it('states direction and weight in the accessible name', () => {
    render(<DivergingWords data={data} title="t" />)
    expect(screen.getByLabelText('fee: toward scam, weight 0.400')).toBeInTheDocument()
    expect(screen.getByLabelText('team: toward genuine, weight -0.200')).toBeInTheDocument()
  })

  it('warns that single words are not proof', () => {
    render(<DivergingWords data={data} title="t" />)
    expect(screen.getByText(/hints, not proof/i)).toBeInTheDocument()
  })

  it('survives an all-zero weight set without dividing by zero', () => {
    render(<DivergingWords data={[{ word: 'a', weight: 0 }]} title="t" />)
    expect(screen.getByLabelText(/^a: toward genuine/)).toBeInTheDocument()
  })
})

describe('LabelDistribution', () => {
  it('shows counts and shares for every verdict', () => {
    render(<LabelDistribution counts={{ Safe: 3, Suspicious: 1, 'Likely Scam': 4 }} />)
    expect(screen.getByText('3 · 38%')).toBeInTheDocument()
    expect(screen.getByText('4 · 50%')).toBeInTheDocument()
  })

  it('pairs every bar with its verdict name, never colour alone', () => {
    render(<LabelDistribution counts={{ Safe: 1, 'Likely Scam': 1 }} />)
    for (const name of ['Safe', 'Suspicious', 'Likely Scam']) {
      expect(screen.getByText(name)).toBeInTheDocument()
    }
  })

  it('reports an empty set instead of dividing by zero', () => {
    render(<LabelDistribution counts={{}} />)
    expect(screen.getByText(/No analyses yet/i)).toBeInTheDocument()
  })
})

describe('CountBars', () => {
  it('renders a row per category with its count', () => {
    render(<CountBars data={[{ name: 'All three', value: 7 }, { name: 'Text only', value: 2 }]} />)
    expect(screen.getByText('All three')).toBeInTheDocument()
    expect(screen.getByText('7')).toBeInTheDocument()
  })

  it('shows the empty message for no data', () => {
    render(<CountBars data={[]} emptyText="Nothing recorded." />)
    expect(screen.getByText('Nothing recorded.')).toBeInTheDocument()
  })
})

describe('DataTable', () => {
  it('provides the table view every chart must have', () => {
    render(<DataTable caption="Per input" head={['Input', 'Value']} rows={[['Text', '93%']]} />)
    expect(screen.getByRole('table', { name: 'Per input' })).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'Input' })).toBeInTheDocument()
    expect(screen.getByRole('cell', { name: '93%' })).toBeInTheDocument()
  })
})

describe('chart/table toggle in the breakdown', () => {
  it('switches between the chart and its table view', async () => {
    const { default: ResultView } = await import('./ResultView')
    const { scamResult } = await import('../test/fixtures')
    render(<ResultView result={scamResult} onReset={() => {}} />)
    await userEvent.click(screen.getByRole('button', { name: 'Show table' }))
    expect(screen.getByRole('table', { name: /scam probability per input/i })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Show chart' }))
    expect(screen.queryByRole('table', { name: /scam probability per input/i })).not.toBeInTheDocument()
  })
})
