import { useEffect, useMemo, useRef, useState } from 'react'
import { analyze, ApiError } from '../api/client'
import type { AnalyzeResponse } from '../api/types'
import { ImageCard, TextCard, UrlCard } from '../components/InputCards'
import LoadingSteps, { stepsFor } from '../components/LoadingSteps'
import ResultView from '../components/ResultView'
import { canSubmit, hasInput, validateForm } from '../lib/validation'

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
    return <div ref={resultRef} tabIndex={-1} className="outline-none" aria-label="Analysis result"><ResultView result={result} onReset={reset} /></div>
  }

  return (
    <form onSubmit={submit} noValidate className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold text-navy-900">Is this job or internship real?</h1>
        <p className="mt-1 text-slate-600">Add any combination of text, an image and a link. The more you add, the better we can cross-check.</p>
      </div>
      {loading ? (
        <LoadingSteps steps={stepsFor({ text: Boolean(text.trim()), image: Boolean(image), url: Boolean(url.trim()) })} onCancel={() => abortRef.current?.abort('cancelled')} />
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <TextCard value={text} onChange={setText} error={errors.text} />
            <ImageCard file={image} onChange={setImage} error={errors.image} />
            <UrlCard value={url} onChange={setUrl} error={errors.url} />
          </div>
          {error && <p role="alert" className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm font-medium text-red-800">{error}</p>}
          <div className="flex items-center gap-3">
            <button type="submit" disabled={!ready}
              className="rounded-lg bg-emerald-600 px-6 py-3 text-base font-semibold text-white shadow-sm hover:bg-emerald-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-300 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500">
              Analyze
            </button>
            {!hasInput(form) && <span className="text-sm text-slate-500">Add at least one input to begin.</span>}
          </div>
        </>
      )}
    </form>
  )
}
