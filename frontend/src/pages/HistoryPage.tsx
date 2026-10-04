import { useEffect, useState } from 'react'
import { ApiError, getHistory } from '../api/client'
import type { HistoryItem } from '../api/types'
import { LABEL_STYLES } from '../lib/reasons'

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    const c = new AbortController()
    getHistory(c.signal).then(setItems).catch((e) => { if (!(e instanceof ApiError && e.code === 'cancelled')) setError(e instanceof ApiError ? e.message : 'Could not load history.') })
    return () => c.abort('cancelled')
  }, [])
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-navy-900">Recent checks</h1>
      <p className="text-sm text-slate-600">The last 20 analyses on this server. Phone numbers, e-mails and ID numbers are masked, and images are never stored.</p>
      {error && <p role="alert" className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-800">{error}</p>}
      {!items && !error && <p role="status" className="text-slate-500">Loading…</p>}
      {items && items.length === 0 && <p className="rounded-lg bg-white p-6 text-center text-slate-500 shadow-sm">No checks yet. Analyze a posting to see it here.</p>}
      <ul className="space-y-3">
        {items?.map((h) => {
          const s = LABEL_STYLES[h.label]
          return (
            <li key={h.id} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className={`rounded-full px-3 py-0.5 text-sm font-semibold ring-1 ${s.bg} ${s.text} ${s.ring}`}>{h.label}</span>
                <span className="text-sm font-semibold text-navy-900">{Math.round(h.risk_score)}% risk</span>
              </div>
              <p className="mt-2 text-xs text-slate-500">{new Date(h.created_at).toLocaleString()} · inputs: {h.inputs_present.replace(/\+/g, ', ')}{h.url_domain ? ` · ${h.url_domain}` : ''}</p>
              {h.text_preview && <p className="mt-2 text-sm text-slate-700">“{h.text_preview}”</p>}
              {h.reasons[0] && <p className="mt-1 text-sm text-slate-600">{h.reasons[0]}</p>}
            </li>
          )
        })}
      </ul>
    </div>
  )
}
