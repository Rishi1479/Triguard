import { describe, expect, it } from 'vitest'
import { parseReason } from './reasons'

describe('parseReason', () => {
  it('extracts severity, source and highlighted evidence', () => {
    const p = parseReason('High risk – Text: Asks you to pay a fee. (“Pay Rs 2000”) Do not pay.')
    expect(p.severity).toBe('high')
    expect(p.source).toBe('Text')
    expect(p.parts.filter((x) => x.evidence).map((x) => x.text)).toEqual(['Pay Rs 2000'])
    expect(p.parts.map((x) => x.text).join('')).toContain('Do not pay.')
  })
  it('maps Warning and Note', () => {
    expect(parseReason('Warning – Link: x').severity).toBe('medium')
    expect(parseReason('Note – Text: y').severity).toBe('low')
  })
  it('handles unprefixed reasons and several quotes', () => {
    const p = parseReason('No scam warning signs were found.')
    expect(p.severity).toBe('low')
    expect(p.parts).toEqual([{ text: 'No scam warning signs were found.', evidence: false }])
    expect(parseReason('Note – Text: words “fee”, “pay”.').parts.filter((x) => x.evidence)).toHaveLength(2)
  })
})
