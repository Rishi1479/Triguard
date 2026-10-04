import { useEffect, useState } from 'react'
import { ApiError, getHistory, getHistoryDetail } from '../api/client'
import type { HistoryDetail, HistoryItem } from '../api/types'
import { Breakdown, CrossModalWarnings, ProvisionalNotice, ReasonList, allFlags } from '../components/ResultView'
import RiskGauge from '../components/RiskGauge'
import { LABEL_STYLES } from '../lib/reasons'

function Badge({ label }: { label: HistoryItem['label'] }) {
  const s = LABEL_STYLES[label]
  return <span className={`rounded-full px-3 py-0.5 text-sm font-semibold ring-1 ${s.bg} ${s.text} ${s.ring}`}>{label}</span>
}

function DetailView({ id, onBack }: { id: number; onBack: () => void }) {
  const [d, setD] = useState<HistoryDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    const c = new AbortController()
    setD(null)
    setError(null)
    getHistoryDetail(id, c.signal).catch((e) => {
      if (!(e instanceof ApiError && e.code === 'cancelled')) setError(e instanceof ApiError ? e.message : 'Could not load this check.')
      return null
    }).then((v) => v && setD(v))
    return () => c.abort('cancelled')
  }, [id])

  return (
    <div className="space-y-4">
      <button type="button" onClick={onBack} className="text-sm font-medium text-navy-700 underline hover:text-navy-900">← Back to recent checks</button>
      {error && <p role="alert" className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-800">{error}</p>}
      {!d && !error && <p role="status" className="text-slate-500">Loading…</p>}
      {d && (
        <>
          <section className="rounded-xl border border-slate-200 bg-white p-5 text-center shadow-sm">
            <Badge label={d.result.label} />
            <div className="mt-4"><RiskGauge score={d.result.risk_score} label={d.result.label} /></div>
            <p className="mt-2 text-xs text-slate-500">
              Check #{d.id} · {new Date(d.created_at).toLocaleString()} · analysed in {d.latency_ms} ms
            </p>
          </section>

          <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm" aria-labelledby="sub-title">
            <h3 id="sub-title" className="mb-2 text-lg font-bold text-navy-900">What was submitted</h3>
            <dl className="space-y-2 text-sm">
              <div>
                <dt className="font-medium text-slate-700">Inputs</dt>
                <dd className="text-slate-600">{d.inputs_present.replace(/\+/g, ', ')}</dd>
              </div>
              {d.submitted_text && (
                <div>
                  <dt className="font-medium text-slate-700">Text (phone numbers, e-mails and ID numbers masked)</dt>
                  <dd className="mt-1 whitespace-pre-wrap rounded bg-slate-50 p-2 text-slate-700">{d.submitted_text}</dd>
                </div>
              )}
              {d.url_domain && (
                <div>
                  <dt className="font-medium text-slate-700">Link</dt>
                  <dd className="text-slate-600">{d.url_domain} <span className="text-xs text-slate-500">(only the domain is stored)</span></dd>
                </div>
              )}
              {d.image_submitted && (
                <div>
                  <dt className="font-medium text-slate-700">Image</dt>
                  <dd className="text-slate-600">An image was analysed. It was never saved — only a one-way hash was kept.</dd>
                </div>
              )}
            </dl>
          </section>

          <ProvisionalNotice result={d.result} />
          <CrossModalWarnings flags={d.result.cross_modal_flags} />

          <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm" aria-labelledby="why2">
            <h3 id="why2" className="text-lg font-bold text-navy-900">Why?</h3>
            <p className="mb-3 text-sm text-slate-600">Each point can be expanded for a plain-English explanation.</p>
            <ReasonList reasons={d.result.reasons} flags={allFlags(d.result)} />
          </section>

          {d.detail_available ? (
            <section aria-labelledby="break2">
              <h3 id="break2" className="mb-2 text-lg font-bold text-navy-900">Breakdown by input</h3>
              <Breakdown results={d.result.modality_results} />
            </section>
          ) : (
            <p className="rounded-lg border border-slate-200 bg-white p-3 text-sm text-slate-600">
              This check was saved before per-input details were recorded, so only the verdict and reasons are available.
            </p>
          )}

          {d.feedback.length > 0 && (
            <section className="rounded-xl border border-slate-200 bg-white p-4 text-sm shadow-sm">
              <h3 className="mb-1 font-bold text-navy-900">Feedback given</h3>
              <ul className="space-y-1 text-slate-700">
                {d.feedback.map((f, i) => (
                  <li key={i}>
                    Marked <strong>{f.correct ? 'correct' : 'incorrect'}</strong> on {new Date(f.created_at).toLocaleString()}
                    {f.comment && <> — “{f.comment}”</>}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  )
}

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<number | null>(null)

  useEffect(() => {
    if (selected !== null) return
    const c = new AbortController()
    getHistory(c.signal).catch((e) => {
      if (!(e instanceof ApiError && e.code === 'cancelled')) setError(e instanceof ApiError ? e.message : 'Could not load history.')
      return null
    }).then((v) => v && setItems(v))
    return () => c.abort('cancelled')
  }, [selected])

  if (selected !== null) return <DetailView id={selected} onBack={() => setSelected(null)} />

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-navy-900">Recent checks</h1>
      <p className="text-sm text-slate-600">
        The last 20 analyses on this server. Select one to see the full report. Phone numbers, e-mails and ID numbers are masked, and images are never stored.
      </p>
      {error && <p role="alert" className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-800">{error}</p>}
      {!items && !error && <p role="status" className="text-slate-500">Loading…</p>}
      {items && items.length === 0 && <p className="rounded-lg bg-white p-6 text-center text-slate-500 shadow-sm">No checks yet. Analyze a posting to see it here.</p>}
      <ul className="space-y-3">
        {items?.map((h) => (
          <li key={h.id}>
            <button
              type="button"
              onClick={() => setSelected(h.id)}
              className="w-full rounded-xl border border-slate-200 bg-white p-4 text-left shadow-sm transition-colors hover:border-navy-300 hover:bg-navy-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-navy-300"
              aria-label={`Open full report for check ${h.id}, ${h.label}, ${Math.round(h.risk_score)} percent risk`}
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <Badge label={h.label} />
                <span className="text-sm font-semibold text-navy-900">{Math.round(h.risk_score)}% risk</span>
              </div>
              <p className="mt-2 text-xs text-slate-500">
                {new Date(h.created_at).toLocaleString()} · inputs: {h.inputs_present.replace(/\+/g, ', ')}{h.url_domain ? ` · ${h.url_domain}` : ''}
              </p>
              {h.text_preview && <p className="mt-2 text-sm text-slate-700">“{h.text_preview}”</p>}
              {h.reasons[0] && <p className="mt-1 text-sm text-slate-600">{h.reasons[0]}</p>}
              <span className="mt-2 inline-block text-xs font-medium text-navy-700 underline">View full report →</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
