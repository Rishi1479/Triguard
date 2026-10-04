import { useEffect, useMemo, useState } from 'react'
import { ApiError, getHistory, getHistoryDetail } from '../api/client'
import type { HistoryDetail, HistoryItem, Label } from '../api/types'
import {
  Breakdown,
  ConfidencePanel,
  CrossModalWarnings,
  ModelEvidence,
  ProvisionalNotice,
  ReasonList,
  VerdictBadge,
  allFlags,
} from '../components/ResultView'
import { RiskBandMeter } from '../components/charts'
import { MODALITY_ICON, SearchIcon } from '../components/Icons'
import RiskGauge from '../components/RiskGauge'
import { Button, Card, EmptyState, ErrorNote, SectionTitle, Skeleton } from '../components/ui'
import { RISK_TOKENS } from '../lib/riskTokens'

const FILTERS: (Label | 'All')[] = ['All', 'Safe', 'Suspicious', 'Likely Scam']

function relativeTime(iso: string): string {
  const then = new Date(iso).getTime()
  const mins = Math.round((Date.now() - then) / 60000)
  if (!Number.isFinite(mins)) return ''
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins} min ago`
  const hrs = Math.round(mins / 60)
  if (hrs < 24) return `${hrs} h ago`
  return `${Math.round(hrs / 24)} d ago`
}

/** Which inputs a check used, from the backend's "text+url" style string. */
function InputIcons({ present }: { present: string }) {
  const parts = present.split('+').filter((p) => p in MODALITY_ICON) as (keyof typeof MODALITY_ICON)[]
  return (
    <span className="flex items-center gap-1" title={`Inputs: ${parts.join(', ')}`}>
      {parts.map((p) => {
        const Icon = MODALITY_ICON[p]
        return <Icon key={p} width={14} height={14} style={{ color: 'var(--ink-muted)' }} />
      })}
      <span className="sr-only">Inputs: {parts.join(', ')}</span>
    </span>
  )
}

function DetailView({ id, onBack }: { id: number; onBack: () => void }) {
  const [d, setD] = useState<HistoryDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const c = new AbortController()
    setD(null)
    setError(null)
    getHistoryDetail(id, c.signal)
      .catch((e) => {
        if (!(e instanceof ApiError && e.code === 'cancelled')) {
          setError(e instanceof ApiError ? e.message : 'Could not load this check.')
        }
        return null
      })
      .then((v) => v && setD(v))
    return () => c.abort('cancelled')
  }, [id])

  return (
    <div className="space-y-4">
      <Button variant="ghost" size="sm" onClick={onBack}>← Back to recent checks</Button>
      {error && <ErrorNote>{error}</ErrorNote>}
      {!d && !error && (
        <div className="space-y-4">
          <Skeleton className="h-44" />
          <Skeleton className="h-28" />
        </div>
      )}
      {d && (
        <>
          <Card className="p-5">
            <div className="flex flex-col items-center gap-5 sm:flex-row sm:items-start">
              <div className="shrink-0"><RiskGauge score={d.result.risk_score} label={d.result.label} /></div>
              <div className="flex-1 text-center sm:text-left">
                <VerdictBadge label={d.result.label} />
                <p className="mt-2 text-xs text-[var(--ink-muted)]">
                  Check #{d.id} · {new Date(d.created_at).toLocaleString()} · analysed in {d.latency_ms} ms
                </p>
                <div className="mt-4"><RiskBandMeter score={d.result.risk_score} label={d.result.label} /></div>
              </div>
            </div>
          </Card>

          <Card className="p-4" aria-labelledby="sub-title">
            <SectionTitle id="sub-title">What was submitted</SectionTitle>
            <dl className="space-y-2 text-sm">
              <div>
                <dt className="font-medium text-[var(--ink-primary)]">Inputs</dt>
                <dd className="text-[var(--ink-secondary)]">{d.inputs_present.replace(/\+/g, ', ')}</dd>
              </div>
              {d.submitted_text && (
                <div>
                  <dt className="font-medium text-[var(--ink-primary)]">
                    Text <span className="font-normal text-[var(--ink-muted)]">(phone numbers, e-mails and ID numbers masked)</span>
                  </dt>
                  <dd
                    className="mt-1 whitespace-pre-wrap rounded p-2 text-[var(--ink-secondary)]"
                    style={{ background: 'var(--surface-sunken)' }}
                  >
                    {d.submitted_text}
                  </dd>
                </div>
              )}
              {d.url_domain && (
                <div>
                  <dt className="font-medium text-[var(--ink-primary)]">Link</dt>
                  <dd className="text-[var(--ink-secondary)]">
                    {d.url_domain} <span className="text-xs text-[var(--ink-muted)]">(only the domain is stored)</span>
                  </dd>
                </div>
              )}
              {d.image_submitted && (
                <div>
                  <dt className="font-medium text-[var(--ink-primary)]">Image</dt>
                  <dd className="text-[var(--ink-secondary)]">
                    An image was analysed. It was never saved — only a one-way hash was kept.
                  </dd>
                </div>
              )}
            </dl>
          </Card>

          <ProvisionalNotice result={d.result} />
          <CrossModalWarnings flags={d.result.cross_modal_flags} />

          <Card className="p-4 sm:p-5" aria-labelledby="why2">
            <SectionTitle id="why2" hint="Each point can be expanded for a plain-English explanation.">Why?</SectionTitle>
            <ReasonList reasons={d.result.reasons} flags={allFlags(d.result)} />
          </Card>

          {d.detail_available ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <Breakdown results={d.result.modality_results} />
              <div className="space-y-4">
                <ConfidencePanel result={d.result} />
                <ModelEvidence results={d.result.modality_results} />
              </div>
            </div>
          ) : (
            <Card className="p-3 text-sm text-[var(--ink-secondary)]">
              This check was saved before per-input details were recorded, so only the verdict and reasons are available.
            </Card>
          )}

          {d.feedback.length > 0 && (
            <Card className="p-4 text-sm">
              <h3 className="mb-1 font-bold text-[var(--ink-primary)]">Feedback given</h3>
              <ul className="space-y-1 text-[var(--ink-secondary)]">
                {d.feedback.map((f, i) => (
                  <li key={i}>
                    Marked <strong>{f.correct ? 'correct' : 'incorrect'}</strong> on {new Date(f.created_at).toLocaleString()}
                    {f.comment && <> — “{f.comment}”</>}
                  </li>
                ))}
              </ul>
            </Card>
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
  const [filter, setFilter] = useState<Label | 'All'>('All')
  const [query, setQuery] = useState('')

  useEffect(() => {
    if (selected !== null) return
    const c = new AbortController()
    getHistory(c.signal)
      .catch((e) => {
        if (!(e instanceof ApiError && e.code === 'cancelled')) {
          setError(e instanceof ApiError ? e.message : 'Could not load history.')
        }
        return null
      })
      .then((v) => v && setItems(v))
    return () => c.abort('cancelled')
  }, [selected])

  const shown = useMemo(() => {
    if (!items) return null
    const q = query.trim().toLowerCase()
    return items.filter((h) => {
      if (filter !== 'All' && h.label !== filter) return false
      if (!q) return true
      return (
        (h.text_preview ?? '').toLowerCase().includes(q) ||
        (h.url_domain ?? '').toLowerCase().includes(q) ||
        h.reasons.some((r) => r.toLowerCase().includes(q))
      )
    })
  }, [items, filter, query])

  if (selected !== null) return <DetailView id={selected} onBack={() => setSelected(null)} />

  const counts = (l: Label | 'All') => (l === 'All' ? items?.length ?? 0 : items?.filter((h) => h.label === l).length ?? 0)

  return (
    <div className="space-y-4">
      <section>
        <h1 className="text-2xl font-bold tracking-tight text-[var(--ink-primary)]">Recent checks</h1>
        <p className="mt-1 text-[var(--ink-secondary)]">
          The last 20 analyses on this server. Phone numbers, e-mails and ID numbers are masked, and images are never stored.
        </p>
      </section>

      {error && <ErrorNote>{error}</ErrorNote>}

      {/* Filters in one row above the list. */}
      {items && items.length > 0 && (
        <Card className="flex flex-wrap items-center gap-3 p-3">
          <div role="group" aria-label="Filter by verdict" className="flex flex-wrap gap-1">
            {FILTERS.map((f) => {
              const active = filter === f
              const tone = f === 'All' ? null : RISK_TOKENS[f]
              return (
                <button
                  key={f}
                  type="button"
                  onClick={() => setFilter(f)}
                  aria-pressed={active}
                  className="rounded-full px-3 py-1 text-xs font-semibold"
                  style={{
                    background: active ? (tone?.wash ?? 'var(--accent)') : 'var(--surface-sunken)',
                    color: active ? (tone?.ink ?? 'var(--accent-ink)') : 'var(--ink-secondary)',
                  }}
                >
                  {f} <span className="font-normal opacity-70">({counts(f)})</span>
                </button>
              )
            })}
          </div>
          <label className="ml-auto flex min-w-[12rem] flex-1 items-center gap-2 rounded-md border px-2 py-1.5 sm:max-w-xs"
            style={{ borderColor: 'var(--line-strong)' }}>
            <SearchIcon width={15} height={15} style={{ color: 'var(--ink-muted)' }} />
            <span className="sr-only">Search checks</span>
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search text, domain or reason"
              className="w-full bg-transparent text-sm text-[var(--ink-primary)] placeholder:text-[var(--ink-muted)] focus:outline-none"
            />
          </label>
        </Card>
      )}

      {!items && !error && (
        <div className="space-y-3">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-28" />)}</div>
      )}

      {items && items.length === 0 && (
        <EmptyState title="No checks yet">Analyze a posting and it will appear here.</EmptyState>
      )}

      {shown && items && items.length > 0 && shown.length === 0 && (
        <EmptyState title="Nothing matches">Try a different verdict filter or search term.</EmptyState>
      )}

      <ul className="space-y-3">
        {shown?.map((h) => (
          <li key={h.id}>
            <button
              type="button"
              onClick={() => setSelected(h.id)}
              className="w-full rounded-[var(--radius-card)] border bg-[var(--surface-card)] p-4 text-left transition-colors hover:bg-[var(--surface-sunken)]"
              style={{ borderColor: 'var(--line-hairline)' }}
              aria-label={`Open full report for check ${h.id}, ${h.label}, ${Math.round(h.risk_score)} percent risk`}
            >
              <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                <VerdictBadge label={h.label} />
                <span className="text-sm font-semibold text-[var(--ink-primary)] tabular">{Math.round(h.risk_score)}% risk</span>
                <span className="ml-auto flex items-center gap-3 text-xs text-[var(--ink-muted)]">
                  <InputIcons present={h.inputs_present} />
                  <span title={new Date(h.created_at).toLocaleString()}>{relativeTime(h.created_at)}</span>
                </span>
              </div>
              {h.url_domain && (
                <p className="mt-2 font-mono text-xs text-[var(--ink-secondary)]">{h.url_domain}</p>
              )}
              {h.text_preview && (
                <p className="mt-2 line-clamp-2 text-sm text-[var(--ink-secondary)]">“{h.text_preview}”</p>
              )}
              {h.reasons[0] && <p className="mt-1 text-sm text-[var(--ink-secondary)]">{h.reasons[0]}</p>}
              <span className="mt-2 inline-block text-xs font-semibold" style={{ color: 'var(--accent)' }}>
                View full report →
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
