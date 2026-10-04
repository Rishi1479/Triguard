import type { Label, Severity } from '../api/types'

export interface ReasonPart {
  text: string
  evidence: boolean
}

export interface ParsedReason {
  severity: Severity
  source: string
  parts: ReasonPart[]
}

const PREFIX: Record<string, Severity> = { 'High risk': 'high', Warning: 'medium', Note: 'low' }

/** Split a backend reason such as `High risk – Text: Asks for a fee (“Pay Rs 2000”)` into severity, source and highlighted evidence. */
export function parseReason(reason: string): ParsedReason {
  const m = /^(High risk|Warning|Note) – ([^:]+): ([\s\S]*)$/.exec(reason)
  const severity: Severity = m ? PREFIX[m[1]] : 'low'
  const source = m ? m[2] : ''
  const body = m ? m[3] : reason
  const parts: ReasonPart[] = []
  const re = /“([^”]+)”/g
  let last = 0
  let hit: RegExpExecArray | null
  while ((hit = re.exec(body))) {
    if (hit.index > last) parts.push({ text: body.slice(last, hit.index), evidence: false })
    parts.push({ text: hit[1], evidence: true })
    last = hit.index + hit[0].length
  }
  if (last < body.length) parts.push({ text: body.slice(last), evidence: false })
  return { severity, source, parts }
}

export const LABEL_STYLES: Record<Label, { text: string; bg: string; ring: string; stroke: string; summary: string }> = {
  Safe: {
    text: 'text-emerald-800',
    bg: 'bg-emerald-100',
    ring: 'ring-emerald-300',
    stroke: '#059669',
    summary: 'We found no strong scam signals. Still verify the employer yourself before sharing personal data.',
  },
  Suspicious: {
    text: 'text-amber-900',
    bg: 'bg-amber-100',
    ring: 'ring-amber-300',
    stroke: '#d97706',
    summary: 'Some warning signs were found. Do not pay or share documents until you have verified this employer.',
  },
  'Likely Scam': {
    text: 'text-red-800',
    bg: 'bg-red-100',
    ring: 'ring-red-300',
    stroke: '#dc2626',
    summary: 'This looks like a scam. Do not pay any money or share personal documents.',
  },
}
