import { useState } from 'react'
import { sendFeedback } from '../api/client'
import type { AnalyzeResponse, Label, ModalityName, ModalityResult, RedFlag } from '../api/types'
import { DataTable, DivergingWords, ModalityBars, RiskBandMeter } from './charts'
import { AlertIcon, InfoIcon, MODALITY_ICON, RISK_ICON, SeverityIcon } from './Icons'
import ReasonDetails from './ReasonDetails'
import RiskGauge from './RiskGauge'
import { Accordion, Button, Card, Chip, SectionTitle } from './ui'
import { LABEL_SUMMARY, RISK_TOKENS, SEVERITY_TOKENS, modalityHasChart } from '../lib/riskTokens'

const MODALITY_TITLE: Record<ModalityName, string> = { text: 'Text', image: 'Image', url: 'Link' }
const SEVERITY_WORD = { high: 'High risk', medium: 'Warning', low: 'Note' } as const

export function VerdictBadge({ label }: { label: Label }) {
  const t = RISK_TOKENS[label]
  const Icon = RISK_ICON[label]
  return (
    <span
      className="risk-chip inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-lg font-bold"
      style={{ background: t.wash, color: t.ink }}
    >
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
    <Card
      role="alert"
      className="p-4"
      style={{ background: 'var(--risk-scam-wash)', borderColor: 'var(--risk-scam)' }}
      aria-labelledby="cross-title"
    >
      <h3 id="cross-title" className="flex items-center gap-2 text-base font-bold" style={{ color: 'var(--risk-scam-ink)' }}>
        <AlertIcon /> Your inputs don’t match each other
      </h3>
      <ul className="mt-2 space-y-2">
        {flags.map((f, i) => (
          <li key={i} className="flex gap-2 text-sm" style={{ color: 'var(--risk-scam-ink)' }}>
            <SeverityIcon severity={f.severity} />
            <span>
              <span className="sr-only">{SEVERITY_WORD[f.severity]}. </span>
              {f.message} {f.evidence && <span className="opacity-80">({f.evidence})</span>}
            </span>
          </li>
        ))}
      </ul>
    </Card>
  )
}

/**
 * Shown when the backend abstained: too little evidence, an input unlike anything in training, or
 * a score too close to the fence to call. The label still renders, flagged as provisional.
 */
export function ProvisionalNotice({ result }: { result: { abstained?: boolean } }) {
  if (!result.abstained) return null
  return (
    <Card
      role="status"
      className="p-4"
      style={{ background: 'var(--risk-suspicious-wash)', borderColor: 'var(--risk-suspicious)' }}
      aria-labelledby="provisional-title"
    >
      <h3
        id="provisional-title"
        className="flex items-center gap-2 text-base font-bold"
        style={{ color: 'var(--risk-suspicious-ink)' }}
      >
        <AlertIcon /> Not enough evidence for a confident verdict
      </h3>
      <p className="mt-2 text-sm" style={{ color: 'var(--risk-suspicious-ink)' }}>
        Treat the result below as provisional. For a firmer answer, add more to go on — the full
        posting text, the application link, or a clearer photo of the offer letter.
      </p>
    </Card>
  )
}

/** Abstention, evidence strength, novelty and which inputs actually contributed. */
export function ConfidencePanel({ result }: { result: AnalyzeResponse }) {
  const text = result.modality_results.find((m) => m.modality === 'text')
  const novelty = typeof text?.explanation?.novelty === 'number' ? (text.explanation.novelty as number) : null
  const ood = Boolean(text?.explanation?.out_of_distribution)
  const used = result.modality_results.filter((m) => m.available).map((m) => MODALITY_TITLE[m.modality])
  const uniqueUsed = [...new Set(used)]

  const rows: { term: string; value: React.ReactNode }[] = [
    {
      term: 'Confidence',
      value: result.abstained
        ? <Chip tone={SEVERITY_TOKENS.medium} icon={<AlertIcon width={13} height={13} />}>Low confidence</Chip>
        : <Chip tone={SEVERITY_TOKENS.low} icon={<InfoIcon width={13} height={13} />}>Normal confidence</Chip>,
    },
    { term: 'Inputs used', value: uniqueUsed.length ? uniqueUsed.join(', ') : 'none' },
  ]
  if (novelty !== null) {
    rows.push({
      term: 'Familiarity',
      value: ood
        ? <span style={{ color: 'var(--risk-suspicious-ink)' }}>Unusual wording — {Math.round(novelty * 100)}% of words unseen in training</span>
        : <span>Typical wording ({Math.round(novelty * 100)}% unseen)</span>,
    })
  }

  return (
    <Card className="p-4">
      <SectionTitle id="conf-title" hint="How much the system is willing to stand behind this verdict.">
        Confidence &amp; coverage
      </SectionTitle>
      <dl className="space-y-2 text-sm">
        {rows.map((r) => (
          <div key={r.term} className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <dt className="w-28 shrink-0 text-xs font-medium uppercase tracking-wide text-[var(--ink-muted)]">{r.term}</dt>
            <dd className="text-[var(--ink-secondary)]">{r.value}</dd>
          </div>
        ))}
      </dl>
      {ood && (
        <p className="mt-3 text-xs text-[var(--ink-secondary)]">
          The model had little comparable training data for this input, so its score was pulled back
          toward the average rather than reported as-is.
        </p>
      )}
    </Card>
  )
}

/** Per-input probabilities: a chart when two or more inputs were measured, stat rows otherwise. */
export function Breakdown({ results }: { results: ModalityResult[] }) {
  const [asTable, setAsTable] = useState(false)
  const byModality: ModalityName[] = ['text', 'image', 'url']
  const data = byModality.map((m) => {
    const all = results.filter((r) => r.modality === m)
    const avail = all.filter((r) => r.available)
    return {
      name: MODALITY_TITLE[m],
      value: avail.length ? Math.max(...avail.map((r) => r.probability)) : null,
      icon: MODALITY_ICON[m],
    }
  })

  return (
    <Card className="p-4">
      <div className="flex items-start justify-between gap-3">
        <SectionTitle id="break-title" hint="Each input scored on its own, before they were combined.">
          Breakdown by input
        </SectionTitle>
        <Button variant="ghost" size="sm" onClick={() => setAsTable((v) => !v)} aria-pressed={asTable}>
          {asTable ? 'Show chart' : 'Show table'}
        </Button>
      </div>

      {asTable ? (
        <DataTable
          caption="Scam probability per input"
          head={['Input', 'Scam probability']}
          rows={data.map((d) => [d.name, d.value === null ? 'not given' : `${Math.round(d.value * 100)}%`])}
        />
      ) : modalityHasChart(data.map((d) => d.value)) ? (
        <ModalityBars data={data} title="Scam probability per input" />
      ) : (
        <ul className="space-y-1.5 text-sm">
          {data.map((d) => (
            <li key={d.name} className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-1.5 text-[var(--ink-secondary)]">
                <d.icon width={14} height={14} /> {d.name}
              </span>
              <span className="font-semibold text-[var(--ink-primary)] tabular">
                {d.value === null ? <span className="font-normal text-[var(--ink-muted)]">not given</span> : `${Math.round(d.value * 100)}%`}
              </span>
            </li>
          ))}
        </ul>
      )}

      <KeyFacts results={results} />

      <div className="mt-4 space-y-2">
        {results.filter((r) => r.available && (r.red_flags.length || Object.keys(r.extracted).length)).map((r, i) => (
          <Accordion
            key={`${r.modality}-${i}`}
            summary={
              <span className="text-sm font-medium text-[var(--ink-primary)]">
                {MODALITY_TITLE[r.modality]} details
                {r.red_flags.length > 0 && (
                  <span className="ml-2 text-xs font-normal text-[var(--ink-muted)]">
                    {r.red_flags.length} warning{r.red_flags.length > 1 ? 's' : ''}
                  </span>
                )}
              </span>
            }
          >
            {r.red_flags.length > 0 && (
              <ul className="mb-2 space-y-1.5">
                {r.red_flags.map((f, j) => (
                  <li key={j} className="flex gap-2">
                    <SeverityIcon severity={f.severity} />
                    <span>
                      <span className="sr-only">{SEVERITY_WORD[f.severity]}. </span>
                      {f.message}
                      {f.evidence && <mark className="ml-1 rounded px-1" style={{ background: 'var(--risk-suspicious-wash)', color: 'var(--ink-primary)' }}>“{f.evidence}”</mark>}
                    </span>
                  </li>
                ))}
              </ul>
            )}
            <ExtractedFacts extracted={r.extracted} />
          </Accordion>
        ))}
      </div>
    </Card>
  )
}

/** The one or two facts worth reading without opening a disclosure. */
function KeyFacts({ results }: { results: ModalityResult[] }) {
  const url = results.find((r) => r.modality === 'url' && r.available)
  const domain = typeof url?.extracted?.registered_domain === 'string' ? url.extracted.registered_domain : null
  const age = typeof url?.extracted?.domain_age_days === 'number' ? url.extracted.domain_age_days : null
  if (!domain) return null
  return (
    <dl className="mt-3 flex flex-wrap items-baseline gap-x-4 gap-y-1 border-t pt-3 text-xs" style={{ borderColor: 'var(--line-hairline)' }}>
      <div className="flex items-baseline gap-1.5">
        <dt className="text-[var(--ink-muted)]">Domain</dt>
        <dd className="font-mono font-semibold text-[var(--ink-primary)]">{domain}</dd>
      </div>
      {age !== null && (
        <div className="flex items-baseline gap-1.5">
          <dt className="text-[var(--ink-muted)]">Registered</dt>
          <dd className="font-semibold text-[var(--ink-primary)]">
            {age} day{age === 1 ? '' : 's'} ago
            {age <= 30 && <span className="ml-1 font-normal" style={{ color: 'var(--risk-scam-ink)' }}>— very new</span>}
          </dd>
        </div>
      )}
    </dl>
  )
}

const FACT_LABELS: Record<string, string> = {
  registered_domain: 'Domain',
  domain_age_days: 'Domain age (days)',
  n_chars: 'Characters read',
  ocr_backend: 'OCR engine',
  emails: 'E-mails found',
  phones: 'Phone numbers found',
  urls: 'Links found',
  companies: 'Companies named',
  qr_links: 'QR code links',
}

function ExtractedFacts({ extracted }: { extracted: Record<string, unknown> }) {
  const shown = Object.entries(extracted)
    .filter(([k, v]) => k in FACT_LABELS && v !== null && v !== undefined && (!Array.isArray(v) || v.length > 0))
    .slice(0, 6)
  if (!shown.length) return null
  return (
    <dl className="grid grid-cols-1 gap-1 text-xs sm:grid-cols-2">
      {shown.map(([k, v]) => (
        <div key={k} className="flex gap-2">
          <dt className="text-[var(--ink-muted)]">{FACT_LABELS[k]}:</dt>
          <dd className="truncate text-[var(--ink-secondary)]">{Array.isArray(v) ? v.join(', ') : String(v)}</dd>
        </div>
      ))}
    </dl>
  )
}

/** LIME words, when the backend supplied them. */
export function ModelEvidence({ results }: { results: ModalityResult[] }) {
  const words = results
    .map((r) => r.explanation?.lime_top_words)
    .find((w): w is { word: string; weight: number }[] => Array.isArray(w) && w.length > 0)
  if (!words) return null
  return (
    <Card className="p-4">
      <SectionTitle id="evid-title" hint="The model’s own view, for transparency.">
        What the model reacted to
      </SectionTitle>
      <DivergingWords data={words.slice(0, 8)} title="Word influence on the score" />
    </Card>
  )
}

export function SafetyTips() {
  return (
    <Card className="p-4" style={{ background: 'var(--surface-sunken)' }}>
      <h3 id="tips-title" className="text-sm font-bold text-[var(--ink-primary)]">If in doubt</h3>
      <ul className="mt-2 space-y-1 text-sm text-[var(--ink-secondary)]">
        <li>• Never pay a fee for a job — real employers never ask candidates for money.</li>
        <li>• Never share Aadhaar, PAN, bank details or an OTP with a recruiter.</li>
        <li>• Find the company’s official site yourself and apply there instead.</li>
        <li>
          • Report fraud at{' '}
          <a className="underline" style={{ color: 'var(--accent)' }} href="https://cybercrime.gov.in" target="_blank" rel="noreferrer">
            cybercrime.gov.in
          </a>{' '}
          or call <strong>1930</strong>.
        </li>
      </ul>
    </Card>
  )
}

export function Disclaimer() {
  return (
    <p className="text-center text-xs text-[var(--ink-muted)]">
      TriGuard assists your judgement — it is not legal advice and not a guarantee. A “Safe” result
      is not an endorsement; always verify the employer independently.
    </p>
  )
}

function Feedback({ analysisId }: { analysisId: number }) {
  const [sent, setSent] = useState<boolean | null>(null)
  const [failed, setFailed] = useState(false)
  const send = async (correct: boolean) => {
    try {
      await sendFeedback(analysisId, correct)
      setSent(correct)
    } catch {
      setFailed(true)
    }
  }
  if (sent !== null) {
    return <p role="status" className="text-sm text-[var(--ink-secondary)]">Thanks — your feedback was recorded.</p>
  }
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="text-sm font-medium text-[var(--ink-primary)]">Was this verdict right?</span>
      <Button variant="secondary" size="sm" onClick={() => send(true)}>Yes</Button>
      <Button variant="secondary" size="sm" onClick={() => send(false)}>No</Button>
      {failed && (
        <span role="alert" className="text-xs" style={{ color: 'var(--risk-scam-ink)' }}>
          Could not send — try again later.
        </span>
      )}
    </div>
  )
}

export default function ResultView({ result, onReset }: { result: AnalyzeResponse; onReset: () => void }) {
  return (
    <div className="space-y-5" aria-live="polite">
      <ProvisionalNotice result={result} />

      <Card className="p-5 sm:p-6" aria-label="Verdict">
        <div className="flex flex-col items-center gap-5 sm:flex-row sm:items-start">
          <div className="shrink-0">
            <RiskGauge score={result.risk_score} label={result.label} />
          </div>
          <div className="flex-1 text-center sm:text-left">
            <VerdictBadge label={result.label} />
            {result.abstained && (
              <p className="mt-2 text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--risk-suspicious-ink)' }}>
                Provisional
              </p>
            )}
            <p className="mt-3 text-[var(--ink-secondary)]">{LABEL_SUMMARY[result.label]}</p>
            <div className="mt-4">
              <RiskBandMeter score={result.risk_score} label={result.label} />
            </div>
          </div>
        </div>
      </Card>

      <CrossModalWarnings flags={result.cross_modal_flags} />

      <Card className="p-4 sm:p-5" aria-labelledby="why-title">
        <SectionTitle id="why-title" hint="Expand any point for a plain-English explanation.">
          Why?
        </SectionTitle>
        <ReasonList reasons={result.reasons} flags={allFlags(result)} />
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Breakdown results={result.modality_results} />
        <div className="space-y-4">
          <ConfidencePanel result={result} />
          <ModelEvidence results={result.modality_results} />
        </div>
      </div>

      <Card className="p-4"><Feedback analysisId={result.id} /></Card>
      <SafetyTips />
      <Disclaimer />

      <div className="flex justify-center">
        <Button onClick={onReset} size="lg">Check another posting</Button>
      </div>
    </div>
  )
}
