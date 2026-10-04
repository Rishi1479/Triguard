import { useId, useState } from 'react'
import type { RedFlag } from '../api/types'
import { helpFor } from '../lib/flagHelp'
import { parseReason } from '../lib/reasons'
import { SeverityIcon } from './Icons'

const SEVERITY_WORD = { high: 'High risk', medium: 'Warning', low: 'Note' } as const

/** One reason with an expandable plain-English explanation (what it means, why it matters, what to do). */
export function ReasonItem({ reason, flags }: { reason: string; flags: RedFlag[] }) {
  const [open, setOpen] = useState(false)
  const id = useId()
  const p = parseReason(reason)
  const help = helpFor(reason, flags)
  return (
    <li className="rounded-lg bg-slate-50">
      <div className="flex gap-3 p-3 text-sm">
        <SeverityIcon severity={p.severity} />
        <div className="min-w-0 flex-1">
          <p>
            <span className="sr-only">{SEVERITY_WORD[p.severity]}. </span>
            {p.source && <strong className="mr-1">{p.source}:</strong>}
            {p.parts.map((part, j) =>
              part.evidence ? (
                <mark key={j} className="rounded bg-yellow-200 px-1 text-slate-900">“{part.text}”</mark>
              ) : (
                <span key={j}>{part.text}</span>
              ),
            )}
          </p>
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            aria-controls={id}
            className="mt-1 text-xs font-medium text-navy-700 underline hover:text-navy-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-navy-300"
          >
            {open ? 'Hide explanation' : 'What does this mean?'}
          </button>
          {open && (
            <div id={id} className="mt-2 space-y-2 rounded-md border border-slate-200 bg-white p-3 text-xs text-slate-700">
              <p className="text-sm font-semibold text-navy-900">{help.title}</p>
              <p><strong>What we found:</strong> {help.what}</p>
              <p><strong>Why it matters:</strong> {help.why}</p>
              <p><strong>What to do:</strong> {help.action}</p>
            </div>
          )}
        </div>
      </div>
    </li>
  )
}

/** All reasons for one analysis. `flags` are every red flag behind them (per-modality + cross-modal). */
export default function ReasonDetails({ reasons, flags }: { reasons: string[]; flags: RedFlag[] }) {
  return (
    <ul className="space-y-2">
      {reasons.map((r, i) => (
        <ReasonItem key={i} reason={r} flags={flags} />
      ))}
    </ul>
  )
}
