import { useEffect, useMemo, useRef, useState } from 'react'
import { analyze, ApiError } from '../api/client'
import type { AnalyzeResponse } from '../api/types'
import { ImageCard, TextCard, UrlCard } from '../components/InputCards'
import { ImageIcon, LinkIcon, SparkIcon, TextIcon } from '../components/Icons'
import LoadingSteps, { stepsFor } from '../components/LoadingSteps'
import ResultView from '../components/ResultView'
import { Button, Card, ErrorNote } from '../components/ui'
import { EXAMPLES } from '../lib/examples'
import { canSubmit, hasInput, validateForm } from '../lib/validation'

/** Which inputs are filled, as a compact checklist so the value of adding more is visible. */
function InputProgress({ text, image, url }: { text: boolean; image: boolean; url: boolean }) {
  const items = [
    { on: text, Icon: TextIcon, label: 'Text' },
    { on: image, Icon: ImageIcon, label: 'Image' },
    { on: url, Icon: LinkIcon, label: 'Link' },
  ]
  const n = items.filter((i) => i.on).length
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      {items.map(({ on, Icon, label }) => (
        <span
          key={label}
          className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-medium"
          style={{
            background: on ? 'var(--risk-safe-wash)' : 'var(--surface-sunken)',
            color: on ? 'var(--risk-safe-ink)' : 'var(--ink-muted)',
          }}
        >
          <Icon width={13} height={13} />
          {label}
          <span aria-hidden="true">{on ? '✓' : '—'}</span>
          <span className="sr-only">{on ? 'provided' : 'not provided'}</span>
        </span>
      ))}
      <span className="text-[var(--ink-muted)]">
        {n === 0 ? 'Add at least one.' : n < 3 ? 'More inputs let us cross-check.' : 'All three — best accuracy.'}
      </span>
    </div>
  )
}

export default function AnalyzePage() {
  const [text, setText] = useState('')
  const [url, setUrl] = useState('')
  const [image, setImage] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const resultRef = useRef<HTMLDivElement>(null)

  const form = { text, url, image }
  const errors = useMemo(() => validateForm(form), [text, url, image]) // eslint-disable-line react-hooks/exhaustive-deps
  const ready = canSubmit(form)

  useEffect(() => () => abortRef.current?.abort(), [])
  useEffect(() => { if (result) resultRef.current?.focus() }, [result])

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!ready || loading) return
    setLoading(true)
    setError(null)
    setResult(null)
    abortRef.current = new AbortController()
    try {
      setResult(await analyze({ text, url, image }, abortRef.current.signal))
    } catch (err) {
      if (err instanceof ApiError && err.code === 'cancelled') return
      setError(err instanceof ApiError ? err.message : 'Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => { setResult(null); setText(''); setUrl(''); setImage(null); setError(null) }

  if (result) {
    return (
      <div ref={resultRef} tabIndex={-1} className="outline-none" aria-label="Analysis result">
        <ResultView result={result} onReset={reset} />
      </div>
    )
  }

  return (
    <form onSubmit={submit} noValidate className="space-y-5">
      <section className="text-center sm:text-left">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--ink-primary)] sm:text-3xl">
          Is this job or internship real?
        </h1>
        <p className="mx-auto mt-1.5 max-w-2xl text-[var(--ink-secondary)] sm:mx-0">
          Paste the message, add a screenshot of the offer letter, or drop in the application link.
          We check all three and explain what we found — we never open the link.
        </p>
      </section>

      {loading ? (
        <LoadingSteps
          steps={stepsFor({ text: Boolean(text.trim()), image: Boolean(image), url: Boolean(url.trim()) })}
          onCancel={() => abortRef.current?.abort('cancelled')}
        />
      ) : (
        <>
          <Card className="p-4">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-[var(--ink-primary)]">
                <SparkIcon width={16} height={16} style={{ color: 'var(--accent)' }} />
                Try an example
              </span>
              <span className="text-xs text-[var(--ink-muted)]">
                The first two are the same job — only the behaviour differs.
              </span>
            </div>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.id}
                  type="button"
                  onClick={() => { setText(ex.text); setUrl(ex.url ?? ''); setImage(null); setError(null) }}
                  className="rounded-lg border p-3 text-left transition-colors hover:bg-[var(--surface-sunken)]"
                  style={{ borderColor: 'var(--line-hairline)' }}
                >
                  <span className="block text-sm font-semibold text-[var(--ink-primary)]">{ex.label}</span>
                  <span className="mt-0.5 block text-xs text-[var(--ink-secondary)]">{ex.blurb}</span>
                </button>
              ))}
            </div>
          </Card>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <TextCard value={text} onChange={setText} error={errors.text} />
            <ImageCard file={image} onChange={setImage} error={errors.image} />
            <div className="lg:col-span-3">
              <UrlCard value={url} onChange={setUrl} error={errors.url} />
            </div>
          </div>

          {error && <ErrorNote>{error}</ErrorNote>}

          <Card className="flex flex-wrap items-center justify-between gap-4 p-4">
            <InputProgress text={Boolean(text.trim())} image={Boolean(image)} url={Boolean(url.trim())} />
            <div className="flex items-center gap-2">
              {hasInput(form) && (
                <Button type="button" variant="ghost" size="md" onClick={reset}>Clear</Button>
              )}
              <Button type="submit" size="lg" disabled={!ready}>Analyze</Button>
            </div>
          </Card>
        </>
      )}
    </form>
  )
}
