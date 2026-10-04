import { useEffect, useState } from 'react'
import { getStats } from '../api/client'
import type { Stats } from '../api/types'

const STEPS = [
  ['Text', 'A language model plus rule checks look for fee requests, urgency, “no interview” promises, WhatsApp-only contact, free e-mail addresses and requests for Aadhaar, PAN or OTPs.'],
  ['Image', 'OCR reads offer letters and posters, then the same text checks run on what it finds. QR codes are decoded (never opened), and UPI payment QR codes are flagged.'],
  ['Link', 'We study the web address itself — look-alike domains such as tcs-hiring.online, odd domain endings, shorteners, and (when available) how new the domain is. We never open the page.'],
  ['Cross-check', 'Scams often don’t add up. We compare the company named, the recruiter’s e-mail, the link and the QR code across your inputs.'],
  ['Fusion', 'Evidence from every input is combined into one calibrated risk score and a verdict: Safe, Suspicious or Likely Scam.'],
] as const

export default function AboutPage() {
  const [stats, setStats] = useState<Stats | null>(null)
  useEffect(() => { const c = new AbortController(); getStats(c.signal).then(setStats).catch(() => setStats(null)); return () => c.abort('cancelled') }, [])
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy-900">How TriGuard works</h1>
      <ol className="space-y-3">
        {STEPS.map(([t, d], i) => (
          <li key={t} className="flex gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-navy-700 font-bold text-white" aria-hidden="true">{i + 1}</span>
            <div><h2 className="font-semibold text-navy-900">{t}</h2><p className="text-sm text-slate-700">{d}</p></div>
          </li>
        ))}
      </ol>
      <section aria-labelledby="stats-title">
        <h2 id="stats-title" className="mb-2 text-lg font-bold text-navy-900">Usage statistics</h2>
        {!stats ? <p className="text-sm text-slate-500">Statistics are unavailable right now.</p> : (
          <dl className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {[
              ['Checks', stats.total_analyses],
              ['Likely scams', stats.label_counts['Likely Scam'] ?? 0],
              ['Avg. risk', stats.avg_risk_score == null ? '–' : `${stats.avg_risk_score}%`],
              ['Rated correct', stats.feedback.accuracy == null ? '–' : `${Math.round(stats.feedback.accuracy * 100)}%`],
            ].map(([k, v]) => (
              <div key={k} className="rounded-xl border border-slate-200 bg-white p-4 text-center shadow-sm">
                <dt className="text-xs text-slate-500">{k}</dt><dd className="text-2xl font-bold text-navy-900">{v}</dd>
              </div>
            ))}
          </dl>
        )}
      </section>
      <section className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-700 shadow-sm">
        <h2 className="mb-1 text-lg font-bold text-navy-900">Privacy &amp; limits</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>Uploaded images are analysed in memory and never saved; only a one-way hash is kept.</li>
          <li>Stored text has phone numbers, e-mails and ID numbers masked.</li>
          <li>Results are estimates and can be wrong. Always verify an employer through its official website.</li>
        </ul>
      </section>
    </div>
  )
}
