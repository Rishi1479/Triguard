/**
 * Inline-SVG charts. No chart library: five simple forms do not justify a dependency.
 *
 * Colour comes from the --viz-* tokens in index.css (validated: all six checks pass on the
 * #ffffff surface). Rules held here: one hue per single-measure series (never a
 * value-ramp on nominal categories), status colours only where they mean status and always with
 * an icon + label, solid hairline grid, 2px surface gaps between fills, 4px rounded data-ends
 * anchored to the baseline, selective direct labels, and a table view for every chart.
 */
import { useId, useState } from 'react'
import type { Label } from '../api/types'
import { RISK_ICON } from './Icons'
import { RISK_TOKENS } from '../lib/riskTokens'

const BAND_STOPS = { safe: 30, scam: 70 } as const // mirrors SAFE_THRESHOLD / SCAM_THRESHOLD

/** Horizontal bar path with 4px rounded corners on the value end only, squared at the baseline. */
function barPath(x: number, y: number, w: number, h: number, r = 4): string {
  const rr = Math.max(0, Math.min(r, w, h / 2))
  if (w <= 0) return ''
  return `M${x},${y} H${x + w - rr} a${rr},${rr} 0 0 1 ${rr},${rr} V${y + h - rr} a${rr},${rr} 0 0 1 -${rr},${rr} H${x} Z`
}

function pct(v: number): string {
  return `${Math.round(v)}%`
}

/* ------------------------------------------------------------------ risk band meter */

/**
 * The 0–100 risk score against the three decision bands.
 *
 * The hero number carries the value; this strip exists to show *where the thresholds are*, which
 * a bare gauge cannot. Bands are separated by a 2px surface gap rather than borders.
 */
export function RiskBandMeter({ score, label }: { score: number; label: Label }) {
  const clamped = Math.min(Math.max(score, 0), 100)
  const bands: { from: number; to: number; name: Label }[] = [
    { from: 0, to: BAND_STOPS.safe, name: 'Safe' },
    { from: BAND_STOPS.safe, to: BAND_STOPS.scam, name: 'Suspicious' },
    { from: BAND_STOPS.scam, to: 100, name: 'Likely Scam' },
  ]
  return (
    <figure className="m-0">
      <svg viewBox="0 0 100 16" className="w-full" preserveAspectRatio="none" role="presentation" aria-hidden="true" style={{ height: 34 }}>
        {bands.map((b) => (
          <rect
            key={b.name}
            x={b.from}
            y={4}
            width={Math.max(0, b.to - b.from - 0.5)} /* 0.5 user-unit ≈ 2px surface gap */
            height={5}
            rx={1.5}
            fill={RISK_TOKENS[b.name].bar}
            opacity={b.name === label ? 1 : 0.28}
          />
        ))}
        {/* Value marker: 2px surface ring so it reads on any band. */}
        <g transform={`translate(${clamped} 0)`}>
          <rect x={-1.1} y={0} width={2.2} height={13} rx={1} fill="var(--viz-surface)" />
          <rect x={-0.55} y={1} width={1.1} height={11} rx={0.5} fill="var(--ink-primary)" />
        </g>
      </svg>
      <div className="flex justify-between text-[11px] text-[var(--ink-muted)] tabular">
        <span>0</span>
        <span>{BAND_STOPS.safe} — Suspicious above</span>
        <span>{BAND_STOPS.scam} — Likely scam above</span>
        <span>100</span>
      </div>
      <figcaption className="mt-1 text-xs text-[var(--ink-secondary)]">
        Risk {pct(clamped)} of 100. The highlighted band is this verdict.
      </figcaption>
    </figure>
  )
}

/* ------------------------------------------------------------------ modality bars */

export interface ModalityDatum {
  name: string
  value: number | null // null = input not provided
  icon?: typeof ShieldIcon
}

/**
 * One measure (P(scam)) across up to three inputs: a single hue for every bar, never a ramp.
 * With fewer than two measured inputs a bar chart would be a one-bar chart, so the caller
 * renders stat rows instead — see `modalityHasChart` in lib/riskTokens.
 */
export function ModalityBars({ data, title }: { data: ModalityDatum[]; title: string }) {
  const [hover, setHover] = useState<string | null>(null)
  const titleId = useId()

  return (
    <figure className="m-0">
      <figcaption id={titleId} className="mb-2 text-sm font-semibold text-[var(--ink-primary)]">{title}</figcaption>
      <div className="space-y-1">
        {data.map((d) => {
          const Icon = d.icon
          const unavailable = d.value === null
          const w = unavailable ? 0 : Math.min(Math.max(d.value as number, 0), 1) * 100
          return (
            <div
              key={d.name}
              className="flex items-center gap-2 rounded-md px-1 py-1"
              style={{ background: hover === d.name ? 'var(--surface-sunken)' : 'transparent' }}
              onMouseEnter={() => setHover(d.name)}
              onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(d.name)}
              onBlur={() => setHover(null)}
              tabIndex={0}
              aria-label={unavailable ? `${d.name}: not provided` : `${d.name}: ${pct(w)} scam probability`}
            >
              <span className="flex w-24 shrink-0 items-center gap-1.5 text-xs font-medium text-[var(--ink-secondary)]">
                {Icon && <Icon width={14} height={14} />}
                {d.name}
              </span>
              <span className="relative flex-1">
                <svg viewBox="0 0 100 10" preserveAspectRatio="none" className="w-full" style={{ height: 10 }} aria-hidden="true">
                  <rect x={0} y={1} width={100} height={8} rx={2} fill="var(--viz-track)" />
                  {!unavailable && <path d={barPath(0, 1, w, 8, 2)} fill="var(--viz-series)" />}
                </svg>
              </span>
              {/* ≤ 3 series, so each is direct-labelled rather than relying on a tooltip alone. */}
              <span className="w-16 shrink-0 text-right text-xs font-semibold text-[var(--ink-primary)] tabular">
                {unavailable ? <span className="font-normal text-[var(--ink-muted)]">not given</span> : pct(w)}
              </span>
            </div>
          )
        })}
      </div>
      <p className="mt-1.5 text-xs text-[var(--ink-muted)]">Each bar is that input’s own scam probability, before fusion.</p>
    </figure>
  )
}

/* ------------------------------------------------------------------ diverging (LIME) */

export interface WordWeight {
  word: string
  weight: number
}

/**
 * Which words pushed the score, and which way. Diverging blue↔red with a neutral gray
 * midpoint: the poles must read as opposite and the midpoint as "nothing".
 */
export function DivergingWords({ data, title }: { data: WordWeight[]; title: string }) {
  const [hover, setHover] = useState<string | null>(null)
  const max = Math.max(...data.map((d) => Math.abs(d.weight)), 1e-6)
  return (
    <figure className="m-0">
      <figcaption className="mb-1 text-sm font-semibold text-[var(--ink-primary)]">{title}</figcaption>
      <div className="mb-2 flex items-center gap-4 text-xs text-[var(--ink-secondary)]">
        <span className="flex items-center gap-1.5">
          <span className="viz-mark inline-block h-2 w-3 rounded-sm" style={{ background: 'var(--viz-pos)' }} />
          pushed toward scam
        </span>
        <span className="flex items-center gap-1.5">
          <span className="viz-mark inline-block h-2 w-3 rounded-sm" style={{ background: 'var(--viz-neg)' }} />
          pushed toward genuine
        </span>
      </div>
      <div className="space-y-1">
        {data.map((d) => {
          const frac = (Math.abs(d.weight) / max) * 50
          const toScam = d.weight > 0
          return (
            <div
              key={d.word}
              className="flex items-center gap-2 rounded-md px-1 py-0.5"
              style={{ background: hover === d.word ? 'var(--surface-sunken)' : 'transparent' }}
              onMouseEnter={() => setHover(d.word)}
              onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(d.word)}
              onBlur={() => setHover(null)}
              tabIndex={0}
              aria-label={`${d.word}: ${toScam ? 'toward scam' : 'toward genuine'}, weight ${d.weight.toFixed(3)}`}
            >
              <span className="w-24 shrink-0 truncate text-right text-xs font-medium text-[var(--ink-secondary)]" title={d.word}>
                {d.word}
              </span>
              <svg viewBox="0 0 100 10" preserveAspectRatio="none" className="flex-1" style={{ height: 10 }} aria-hidden="true">
                {/* Neutral gray midpoint rule — solid hairline, never dashed. */}
                <rect x={49.8} y={0} width={0.4} height={10} fill="var(--viz-axis)" />
                {toScam ? (
                  <path d={barPath(50.5, 1, frac, 8, 2)} fill="var(--viz-pos)" />
                ) : (
                  <g transform="translate(100 0) scale(-1 1)">
                    <path d={barPath(50.5, 1, frac, 8, 2)} fill="var(--viz-neg)" />
                  </g>
                )}
              </svg>
            </div>
          )
        })}
      </div>
      <p className="mt-1.5 text-xs text-[var(--ink-muted)]">
        Single words are hints, not proof — they show what the model reacted to.
      </p>
    </figure>
  )
}

/* ------------------------------------------------------------------ label distribution */

/**
 * Verdict counts. These ARE states, so the status palette applies — and every bar carries its
 * icon and name, because red↔green sit at CVD ΔE 4.1 and colour alone would not distinguish them.
 */
export function LabelDistribution({ counts }: { counts: Partial<Record<Label, number>> }) {
  const order: Label[] = ['Safe', 'Suspicious', 'Likely Scam']
  const total = order.reduce((s, l) => s + (counts[l] ?? 0), 0)
  if (!total) return <p className="text-sm text-[var(--ink-secondary)]">No analyses yet.</p>
  return (
    <figure className="m-0 space-y-2">
      {order.map((l) => {
        const n = counts[l] ?? 0
        const share = (n / total) * 100
        const Icon = RISK_ICON[l]
        return (
          <div key={l} className="flex items-center gap-2">
            <span className="flex w-32 shrink-0 items-center gap-1.5 text-xs font-medium text-[var(--ink-secondary)]">
              <Icon width={14} height={14} style={{ color: RISK_TOKENS[l].bar }} />
              {l}
            </span>
            <svg viewBox="0 0 100 10" preserveAspectRatio="none" className="flex-1" style={{ height: 10 }} aria-hidden="true">
              <rect x={0} y={1} width={100} height={8} rx={2} fill="var(--viz-track)" />
              <path d={barPath(0, 1, share, 8, 2)} fill={RISK_TOKENS[l].bar} />
            </svg>
            <span className="w-20 shrink-0 text-right text-xs font-semibold text-[var(--ink-primary)] tabular">
              {n} · {pct(share)}
            </span>
          </div>
        )
      })}
    </figure>
  )
}

/* ------------------------------------------------------------------ simple count bars */

/** Magnitude across nominal categories: one hue for every bar. */
export function CountBars({ data, emptyText = 'No data yet.' }: {
  data: { name: string; value: number }[]
  emptyText?: string
}) {
  const max = Math.max(...data.map((d) => d.value), 1)
  if (!data.length) return <p className="text-sm text-[var(--ink-secondary)]">{emptyText}</p>
  return (
    <figure className="m-0 space-y-2">
      {data.map((d) => (
        <div key={d.name} className="flex items-center gap-2">
          <span className="w-32 shrink-0 truncate text-xs font-medium text-[var(--ink-secondary)]" title={d.name}>{d.name}</span>
          <svg viewBox="0 0 100 10" preserveAspectRatio="none" className="flex-1" style={{ height: 10 }} aria-hidden="true">
            <rect x={0} y={1} width={100} height={8} rx={2} fill="var(--viz-track)" />
            <path d={barPath(0, 1, (d.value / max) * 100, 8, 2)} fill="var(--viz-series)" />
          </svg>
          <span className="w-10 shrink-0 text-right text-xs font-semibold text-[var(--ink-primary)] tabular">{d.value}</span>
        </div>
      ))}
    </figure>
  )
}

/* ------------------------------------------------------------------ table view */

/** The table view every chart must have available. */
export function DataTable({ caption, head, rows }: { caption: string; head: string[]; rows: (string | number)[][] }) {
  return (
    <table className="w-full text-left text-xs">
      <caption className="sr-only">{caption}</caption>
      <thead>
        <tr className="text-[var(--ink-muted)]">
          {head.map((h) => <th key={h} scope="col" className="py-1 pr-3 font-medium">{h}</th>)}
        </tr>
      </thead>
      <tbody className="text-[var(--ink-secondary)]">
        {rows.map((r, i) => (
          <tr key={i} className="border-t" style={{ borderColor: 'var(--line-hairline)' }}>
            {r.map((c, j) => (
              <td key={j} className={`py-1 pr-3 ${j > 0 ? 'tabular' : ''}`}>{c}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
