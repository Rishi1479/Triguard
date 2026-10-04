/** Risk/severity colour tokens. Kept out of component files so fast refresh stays intact. */
import type { Label, Severity } from '../api/types'

export interface Tone {
  bar: string
  wash: string
  ink: string
}

export const RISK_TOKENS: Record<Label, Tone> = {
  Safe: { bar: 'var(--risk-safe)', wash: 'var(--risk-safe-wash)', ink: 'var(--risk-safe-ink)' },
  Suspicious: { bar: 'var(--risk-suspicious)', wash: 'var(--risk-suspicious-wash)', ink: 'var(--risk-suspicious-ink)' },
  'Likely Scam': { bar: 'var(--risk-scam)', wash: 'var(--risk-scam-wash)', ink: 'var(--risk-scam-ink)' },
}

export const SEVERITY_TOKENS: Record<Severity, Tone> = {
  high: RISK_TOKENS['Likely Scam'],
  medium: RISK_TOKENS.Suspicious,
  low: { bar: 'var(--viz-series)', wash: 'var(--surface-sunken)', ink: 'var(--ink-secondary)' },
}

/** Plain-English verdict summaries. */
export const LABEL_SUMMARY: Record<Label, string> = {
  Safe: 'We found no strong scam signals. Still verify the employer yourself before sharing personal data.',
  Suspicious: 'Some warning signs were found. Do not pay or share documents until you have verified this employer.',
  'Likely Scam': 'This looks like a scam. Do not pay any money or share personal documents.',
}

/** True when enough inputs were measured for a bar chart; one bar is a stat tile, not a chart. */
export function modalityHasChart(values: (number | null)[]): boolean {
  return values.filter((v) => v !== null).length >= 2
}
