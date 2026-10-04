import { useState } from 'react'
import { sendFeedback } from '../api/client'
import type { AnalyzeResponse, Label, ModalityName, ModalityResult, RedFlag } from '../api/types'
import ReasonDetails from './ReasonDetails'
import { LABEL_STYLES } from '../lib/reasons'
import { AlertIcon, ShieldIcon, SeverityIcon, XCircleIcon } from './Icons'
import RiskGauge from './RiskGauge'

const MODALITY_TITLE: Record<ModalityName, string> = { text: 'Text analysis', image: 'Image analysis', url: 'Link analysis' }
const SEVERITY_WORD = { high: 'High risk', medium: 'Warning', low: 'Note' } as const
const LOW_BAND = 0.3
const HIGH_BAND = 0.7

export function VerdictBadge({ label }: { label: Label }) {
  const s = LABEL_STYLES[label]
  const Icon = label === 'Safe' ? ShieldIcon : label === 'Suspicious' ? AlertIcon : XCircleIcon
  return (
    <span className={`inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-lg font-bold ring-1 ${s.bg} ${s.text} ${s.ring}`}>
      <Icon /> {label}
    </span>
  )
}

/** All red flags behind a result, used to attach explanations to each reason. */
export function allFlags(result: { modality_results: ModalityResult[]; cross_modal_flags: RedFlag[] }): RedFlag[] {
  return [...result.cross_modal_flags, ...result.modality_results.flatMap((m) => m.red_flags)]
}

export function ReasonList({ reasons, flags = [] }: { reasons: string[]; flags?: RedFlag[] }) {
  return <ReasonDetails reasons={reasons} flags={flags} />
}

export function CrossModalWarnings({ flags }: { flags: RedFlag[] }) {
  if (!flags.length) return null
  return (
    <section role="alert" className="rounded-xl border-2 border-red-300 bg-red-50 p-4" aria-labelledby="cross-title">
      <h3 id="cross-title" className="flex items-center gap-2 text-base font-bold text-red-800">
        <AlertIcon /> Your inputs don’t match each other
      </h3>
      <ul className="mt-2 space-y-2">
        {flags.map((f, i) => (
          <li key={i} className="flex gap-2 text-sm text-red-900">
            <SeverityIcon severity={f.severity} />
            <span>
              <span className="sr-only">{SEVERITY_WORD[f.severity]}. </span>
              {f.message} {f.evidence && <span className="text-red-700">({f.evidence})</span>}
            </span>
          </li>
        ))}
      </ul>
    </section>
  )
}

function ProbabilityBar({ p }: { p: number }) {
  const color = p >= HIGH_BAND ? 'bg-red-500' : p >= LOW_BAND ? 'bg-amber-500' : 'bg-emerald-500'
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200" role="presentation">
      <div className={`h-full ${color}`} style={{ width: `${Math.round(p * 100)}%` }} />
    </div>
  )
}

function ModalityCard({ title, results }: { title: string; results: ModalityResult[] }) {
  const avail = results.filter((r) => r.available)
  if (!avail.length) {
    return (
      <div className="rounded-xl border border-dashed border-slate-300 bg-white p-4 text-sm text-slate-500">
        <h4 className="font-semibold text-slate-700">{title}</h4>
        <p>Not provided</p>
      </div>
    )
  }
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h4 className="mb-2 font-semibold text-navy-900">{title}</h4>
      <div className="space-y-4">
        {avail.map((r, i) => {
          const words = (r.explanation.lime_top_words as { word: string; weight: number }[] | undefined)?.filter((w) => w.weight > 0).slice(0, 5)
          const domain = r.extracted.registered_domain as string | undefined
          const age = r.extracted.domain_age_days as number | null | undefined
          const found = r.extracted.source === 'found_in_content'
          const qr = (r.extracted.qr_links as string[] | undefined) ?? []
          return (
            <div key={i} className="text-sm">
              {domain && <p className="font-medium text-slate-800">{domain}{found && <span className="ml-2 rounded bg-slate-100 px-1.5 py-0.5 text-xs font-normal text-slate-600">found inside your content</span>}</p>}
              <div className="mb-1 flex items-center justify-between text-xs text-slate-600">
                <span>Risk from this input</span><span className="font-semibold">{Math.round(r.probability * 100)}%</span>
              </div>
              <ProbabilityBar p={r.probability} />
              {age != null && <p className="mt-1 text-xs text-slate-600">Domain age: {age} days</p>}
              {qr.length > 0 && <p className="mt-1 break-all text-xs text-slate-600">QR code link: {qr[0]}</p>}
              {r.red_flags.length > 0 && (
                <p className="mt-2 text-xs text-slate-700">{r.red_flags.length} warning sign{r.red_flags.length > 1 ? 's' : ''}: {r.red_flags.map((f) => f.code.replace(/_/g, ' ').toLowerCase()).slice(0, 4).join(', ')}</p>
              )}
              {words && words.length > 0 && (
                <p className="mt-2 text-xs text-slate-700">Most influential words: {words.map((w) => <span key={w.word} className="mr-1 rounded bg-yellow-100 px-1">{w.word}</span>)}</p>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function Breakdown({ results }: { results: ModalityResult[] }) {
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
      {(['text', 'image', 'url'] as const).map((m) => (
        <ModalityCard key={m} title={MODALITY_TITLE[m]} results={results.filter((r) => r.modality === m)} />
      ))}
    </div>
  )
}

export function Feedback({ analysisId }: { analysisId: number }) {
  const [state, setState] = useState<'idle' | 'sending' | 'done' | 'error'>('idle')
  const send = async (correct: boolean) => {
    setState('sending')
    try {
      await sendFeedback(analysisId, correct)
      setState('done')
    } catch {
      setState('error')
    }
  }
  if (state === 'done') return <p role="status" className="text-sm font-medium text-emerald-700">Thanks — your feedback helps us improve.</p>
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="text-sm font-medium text-slate-700">Was this correct?</span>
      <button type="button" disabled={state === 'sending'} onClick={() => send(true)} className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium hover:bg-slate-50 disabled:opacity-50">Yes</button>
      <button type="button" disabled={state === 'sending'} onClick={() => send(false)} className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium hover:bg-slate-50 disabled:opacity-50">No</button>
      {state === 'error' && <span role="alert" className="text-sm text-red-700">Could not send feedback. Please try again.</span>}
    </div>
  )
}

export function SafetyTips() {
  return (
    <section className="rounded-xl border border-navy-100 bg-navy-50 p-4" aria-labelledby="tips-title">
      <h3 id="tips-title" className="mb-2 text-base font-bold text-navy-900">Stay safe</h3>
      <ul className="list-disc space-y-1 pl-5 text-sm text-slate-800">
        <li><strong>Never pay</strong> a registration, training, security or “refundable” fee to get a job or internship.</li>
        <li>Verify the opening on the company’s <strong>official careers page</strong> — type the address yourself, don’t use links from messages.</li>
        <li>Don’t share Aadhaar, PAN, bank details or OTPs before a verified offer.</li>
        <li>Lost money or targeted? Report at <a className="font-medium text-navy-700 underline" href="https://cybercrime.gov.in" target="_blank" rel="noopener noreferrer">cybercrime.gov.in</a> or call <strong>1930</strong>.</li>
      </ul>
    </section>
  )
}

export function Disclaimer() {
  return (
    <p className="text-xs text-slate-500">
      TriGuard is an automated decision-support tool, not legal advice. It can be wrong — a “Safe” result does not guarantee an offer is genuine,
      and a “Likely Scam” result is not proof. Always verify with the employer through official channels.
    </p>
  )
}

export default function ResultView({ result, onReset }: { result: AnalyzeResponse; onReset: () => void }) {
  const style = LABEL_STYLES[result.label]
  return (
    <div className="space-y-5" aria-live="polite">
      <section className="rounded-xl border border-slate-200 bg-white p-5 text-center shadow-sm" aria-label="Verdict">
        <VerdictBadge label={result.label} />
        <div className="mt-4"><RiskGauge score={result.risk_score} label={result.label} /></div>
        <p className="mx-auto mt-3 max-w-lg text-sm text-slate-700">{style.summary}</p>
      </section>
      <CrossModalWarnings flags={result.cross_modal_flags} />
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm" aria-labelledby="why-title">
        <h3 id="why-title" className="text-lg font-bold text-navy-900">Why?</h3>
        <p className="mb-3 text-sm text-slate-600">Each point below can be expanded for a plain-English explanation.</p>
        <ReasonList reasons={result.reasons} flags={allFlags(result)} />
      </section>
      <section aria-labelledby="break-title">
        <h3 id="break-title" className="mb-2 text-lg font-bold text-navy-900">Breakdown by input</h3>
        <Breakdown results={result.modality_results} />
      </section>
      <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"><Feedback analysisId={result.id} /></section>
      <SafetyTips />
      <Disclaimer />
      <button type="button" onClick={onReset} className="rounded-md bg-navy-700 px-4 py-2 text-sm font-semibold text-white hover:bg-navy-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-navy-300">
        Check another posting
      </button>
    </div>
  )
}
