import { useEffect, useId, useRef, useState } from 'react'
import { MAX_TEXT_CHARS } from '../lib/validation'
import { UploadIcon } from './Icons'

const card = 'rounded-[var(--radius-card)] border p-4'
const cardStyle = { borderColor: 'var(--line-hairline)', background: 'var(--surface-card)' }
const errorText = 'mt-2 text-sm font-medium'
const errorStyle = { color: 'var(--risk-scam-ink)' }
const field =
  'w-full rounded-lg border p-3 text-sm bg-transparent text-[var(--ink-primary)] placeholder:text-[var(--ink-muted)] focus:outline-none'
const fieldStyle = { borderColor: 'var(--line-strong)' }

export function TextCard({ value, onChange, error }: { value: string; onChange: (v: string) => void; error: string | null }) {
  const id = useId()
  return (
    <section className={`${card} lg:col-span-2`} style={cardStyle}>
      <label htmlFor={id} className="block text-base font-semibold text-[var(--ink-primary)]">1. Job or internship text</label>
      <p className="mb-2 text-sm text-[var(--ink-secondary)]">Paste the message, advertisement or offer letter text.</p>
      <textarea
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={7}
        aria-invalid={Boolean(error)}
        aria-describedby={error ? `${id}-err` : undefined}
        placeholder="e.g. Congratulations! You are selected. Pay Rs 2,500 registration fee to confirm your seat…"
        className={`${field} resize-y`}
        style={fieldStyle}
      />
      <div className="mt-1 flex justify-between text-xs text-[var(--ink-muted)]">
        <span>{error ? '' : 'Phone numbers and e-mails are masked before anything is stored.'}</span>
        <span>{value.length.toLocaleString()} / {MAX_TEXT_CHARS.toLocaleString()}</span>
      </div>
      {error && <p id={`${id}-err`} role="alert" className={errorText} style={errorStyle}>{error}</p>}
    </section>
  )
}

export function UrlCard({ value, onChange, error }: { value: string; onChange: (v: string) => void; error: string | null }) {
  const id = useId()
  return (
    <section className={card} style={cardStyle}>
      <label htmlFor={id} className="block text-base font-semibold text-[var(--ink-primary)]">3. Application link</label>
      <p className="mb-2 text-sm text-[var(--ink-secondary)]">We only look up the address — we never open the page.</p>
      <input
        id={id}
        type="text"
        inputMode="url"
        autoComplete="off"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-invalid={Boolean(error)}
        aria-describedby={error ? `${id}-err` : undefined}
        placeholder="https://company-careers.example/apply"
        className={field}
        style={fieldStyle}
      />
      {error && <p id={`${id}-err`} role="alert" className={errorText} style={errorStyle}>{error}</p>}
    </section>
  )
}

export function ImageCard({ file, onChange, error }: { file: File | null; onChange: (f: File | null) => void; error: string | null }) {
  const id = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)

  useEffect(() => {
    if (!file || typeof URL.createObjectURL !== 'function') {
      setPreview(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL?.(url)
  }, [file])

  const pick = (f: File | undefined | null) => {
    if (f) onChange(f)
  }

  return (
    <section className={card} style={cardStyle}>
      <h2 className="text-base font-semibold text-[var(--ink-primary)]">2. Offer letter or poster image</h2>
      <p className="mb-2 text-sm text-[var(--ink-secondary)]">PNG, JPG or WebP, up to 5 MB. Images are never saved.</p>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files?.[0]) }}
        className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-4 text-center transition-colors"
        style={{
          borderColor: dragging ? 'var(--accent)' : 'var(--line-strong)',
          background: dragging ? 'var(--viz-track)' : 'var(--surface-sunken)',
        }}
      >
        {preview && file ? (
          <img src={preview} alt={`Preview of ${file.name}`} className="mb-2 max-h-40 rounded border object-contain" style={{ borderColor: 'var(--line-hairline)' }} />
        ) : (
          <UploadIcon className="mb-1" style={{ color: 'var(--ink-muted)' }} width={28} height={28} />
        )}
        <p className="text-sm text-[var(--ink-secondary)]">{file ? file.name : 'Drag & drop an image here, or'}</p>
        <input
          ref={inputRef}
          id={id}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="sr-only"
          aria-label="Choose an image file"
          aria-describedby={error ? `${id}-err` : undefined}
          onChange={(e) => pick(e.target.files?.[0])}
        />
        <div className="mt-2 flex gap-2">
          <button type="button" onClick={() => inputRef.current?.click()}
            className="rounded-md px-3 py-1.5 text-sm font-semibold"
            style={{ background: 'var(--accent)', color: 'var(--accent-ink)' }}>
            {file ? 'Change image' : 'Choose image'}
          </button>
          {file && (
            <button type="button" onClick={() => { onChange(null); if (inputRef.current) inputRef.current.value = '' }}
              className="rounded-md border px-3 py-1.5 text-sm font-medium text-[var(--ink-secondary)]"
              style={{ borderColor: 'var(--line-strong)' }}>
              Remove
            </button>
          )}
        </div>
      </div>
      {error && <p id={`${id}-err`} role="alert" className={errorText} style={errorStyle}>{error}</p>}
    </section>
  )
}
