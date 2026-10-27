import { useEffect, useState } from 'react'
import { ApiError, getStats } from '../api/client'
import type { Stats } from '../api/types'
import { CountBars, DataTable, LabelDistribution } from '../components/charts'
import { Button, Card, EmptyState, ErrorNote, SectionTitle, Skeleton, StatTile } from '../components/ui'

const COMBO_LABEL: Record<string, string> = {
  text: 'Text only',
  image: 'Image only',
  url: 'Link only',
  'text+image': 'Text + image',
  'text+url': 'Text + link',
  'image+url': 'Image + link',
  'text+image+url': 'All three',
}

function prettyCombo(key: string): string {
  return COMBO_LABEL[key] ?? key
}

export default function StatsPage() {
  const [data, setData] = useState<Stats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [asTable, setAsTable] = useState(false)

  useEffect(() => {
    const ac = new AbortController()
    getStats(ac.signal)
      .then(setData)
      .catch((e) => {
        if (e instanceof ApiError && e.code === 'cancelled') return
        setError(e instanceof ApiError ? e.message : 'Could not load statistics.')
      })
      .finally(() => setLoading(false))
    return () => ac.abort()
  }, [])

  if (loading) {
    return (
      <div className="space-y-5">
        <Skeleton className="h-8 w-56" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-24" />)}
        </div>
        <Skeleton className="h-48" />
      </div>
    )
  }

  if (error) return <ErrorNote>{error}</ErrorNote>
  if (!data) return <EmptyState title="No statistics available." />
  if (!data.total_analyses) {
    return (
      <EmptyState title="Nothing analysed yet">
        Run a check on the Analyze page and the numbers will appear here.
      </EmptyState>
    )
  }

  const combos = Object.entries(data.input_combinations)
    .map(([k, v]) => ({ name: prettyCombo(k), value: v }))
    .sort((a, b) => b.value - a.value)

  const fb = data.feedback

  return (
    <div className="space-y-5">
      <section>
        <h1 className="text-2xl font-bold tracking-tight text-[var(--ink-primary)]">Usage statistics</h1>
        <p className="mt-1 text-[var(--ink-secondary)]">
          Aggregates from this instance’s saved checks. Text is masked before storage.
        </p>
      </section>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label="Checks run" value={data.total_analyses.toLocaleString()} tabular />
        <StatTile
          label="Average risk"
          value={data.avg_risk_score ?? '—'}
          unit={data.avg_risk_score === null ? undefined : '%'}
          tabular
        />
        <StatTile
          label="Average response"
          value={data.avg_latency_ms ?? '—'}
          unit={data.avg_latency_ms === null ? undefined : 'ms'}
          hint="Mean end-to-end latency"
          tabular
        />
        <StatTile
          label="Rated correct"
          value={fb.accuracy === null ? '—' : `${Math.round(fb.accuracy * 100)}%`}
          hint={fb.total ? `${fb.marked_correct} of ${fb.total} ratings` : 'No ratings yet'}
          tabular
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="p-4">
          <div className="flex items-start justify-between gap-3">
            <SectionTitle id="dist-title" hint="How verdicts have been distributed.">Verdicts</SectionTitle>
            <Button variant="ghost" size="sm" onClick={() => setAsTable((v) => !v)} aria-pressed={asTable}>
              {asTable ? 'Show chart' : 'Show table'}
            </Button>
          </div>
          {asTable ? (
            <DataTable
              caption="Verdict counts"
              head={['Verdict', 'Count']}
              rows={(['Safe', 'Suspicious', 'Likely Scam'] as const).map((l) => [l, data.label_counts[l] ?? 0])}
            />
          ) : (
            <LabelDistribution counts={data.label_counts} />
          )}
        </Card>

        <Card className="p-4">
          <SectionTitle id="combo-title" hint="Which inputs people actually provide.">Inputs used</SectionTitle>
          <CountBars data={combos} emptyText="No checks recorded yet." />
        </Card>
      </div>

      <Card className="p-4" style={{ background: 'var(--surface-sunken)' }}>
        <p className="text-xs text-[var(--ink-secondary)]">
          These are usage counts, not model accuracy. Measured model performance lives in
          <span className="font-mono"> backend/reports/RESULTS.md</span>, computed on EMSCAD (text) and
          Kaggle malicious_phish (URL).
        </p>
      </Card>
    </div>
  )
}
