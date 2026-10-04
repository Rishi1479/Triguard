import type { AnalyzeResponse } from '../api/types'

export const scamResult: AnalyzeResponse = {
  id: 7,
  latency_ms: 420,
  label: 'Likely Scam',
  risk_score: 87.4,
  reasons: [
    'High risk – Mismatch between inputs: The posting names TCS but links to tcs-hiring.online, not its official site (tcs.com). (“tcs-hiring.online”)',
    'High risk – Text: Asks you to pay a fee or deposit — genuine employers never charge candidates. (“Pay Rs 2000 registration fee”)',
    'Warning – Link: Domain imitates TCS.',
    'Note – Text: the model was most influenced by the words “fee”, “pay”.',
  ],
  cross_modal_flags: [
    { code: 'COMPANY_URL_MISMATCH', message: 'The posting names TCS but links to tcs-hiring.online.', severity: 'high', evidence: 'tcs-hiring.online' },
  ],
  modality_results: [
    { modality: 'text', probability: 0.93, available: true, extracted: {}, explanation: { lime_top_words: [{ word: 'fee', weight: 0.4 }, { word: 'job', weight: -0.1 }] },
      red_flags: [{ code: 'PAYMENT_REQUEST', message: 'Asks you to pay', severity: 'high', evidence: 'Pay Rs 2000' }] },
    { modality: 'image', probability: 0, available: false, extracted: {}, explanation: {}, red_flags: [] },
    { modality: 'url', probability: 0.85, available: true, red_flags: [], explanation: {},
      extracted: { registered_domain: 'tcs-hiring.online', domain_age_days: 6 } },
  ],
}

export const safeResult: AnalyzeResponse = {
  ...scamResult, id: 8, label: 'Safe', risk_score: 8, cross_modal_flags: [],
  reasons: ['No scam warning signs were found in what you provided. Still verify the employer on its official website.'],
}
