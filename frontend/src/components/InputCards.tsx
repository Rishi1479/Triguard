import { useEffect, useId, useRef, useState } from 'react'
import { MAX_TEXT_CHARS } from '../lib/validation'
import { UploadIcon } from './Icons'

const card = 'rounded-xl border border-slate-200 bg-white p-4 shadow-sm'
const errorText = 'mt-2 text-sm font-medium text-red-700'

export function TextCard({ value, onChange, error }: { value: string; onChange: (v: string) => void; error: string | null }) {
  const id = useId()
  return (
    <section className={`${card} lg:col-span-2`}>
      <label htmlFor={id} className="block text-base font-semibold text-navy-900">1. Job or internship text</label>
      <p className="mb-2 text-sm text-slate-600">Paste the message, advertisement or offer letter text.</p>
      <textarea
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={7}
        aria-invalid={Boolean(error)}
        aria-describedby={error ? `${id}-err` : undefined}
        placeholder="e.g. Congratulations! You are selected. Pay Rs 2,500 registration fee to confirm your seat…"
        className="w-full resize-y rounded-lg border border-slate-300 p-3 text-sm focus:border-navy-500 focus:outline-none focus:ring-2 focus:ring-navy-300"
      />
      <div className="mt-1 flex justify-between text-xs text-slate-500">
        <span>{error ? '' : 'Phone numbers and e-mails are masked before anything is stored.'}</span>
        <span>{value.length.toLocaleString()} / {MAX_TEXT_CHARS.toLocaleString()}</span>
      </div>
      {error && <p id={`${id}-err`} role="alert" className={errorText}>{error}</p>}
    </section>
  )
}

export function UrlCard({ value, onChange, error }: { value: string; onChange: (v: string) => void; error: string | null }) {
  const id = useId()
  return (
    <section className={card}>
      <label htmlFor={id} className="block text-base font-semibold text-navy-900">3. Application link</label>
      <p className="mb-2 text-sm text-slate-600">We only look up the address — we never open the page.</p>
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
        className="w-full rounded-lg border border-slate-300 p-3 text-sm focus:border-navy-500 focus:outline-none focus:ring-2 focus:ring-navy-300"
      />
      {error && <p id={`${id}-err`} role="alert" className={errorText}>{error}</p>}
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
    <section className={card}>
      <h2 className="text-base font-semibold text-navy-900">2. Offer letter or poster image</h2>
      <p className="mb-2 text-sm text-slate-600">PNG, JPG or WebP, up to 5 MB. Images are never saved.</p>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files?.[0]) }}
        className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-4 text-center transition-colors ${
          dragging ? 'border-navy-500 bg-navy-50' : 'border-slate-300 bg-slate-50'
        }`}
      >
        {preview && file ? (
          <img src={preview} alt={`Preview of ${file.name}`} className="mb-2 max-h-40 rounded border border-slate-200 object-contain" />
        ) : (
          <UploadIcon className="mb-1 text-slate-400" width={28} height={28} />
        )}
        <p className="text-sm text-slate-600">{file ? file.name : 'Drag & drop an image here, or'}</p>
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
            className="rounded-md bg-navy-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-navy-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-navy-300">
            {file ? 'Change image' : 'Choose image'}
          </button>
          {file && (
            <button type="button" onClick={() => { onChange(null); if (inputRef.current) inputRef.current.value = '' }}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-white focus:outline-none focus-visible:ring-2 focus-visible:ring-navy-300">
              Remove
            </button>
          )}
        </div>
      </div>
      {error && <p id={`${id}-err`} role="alert" className={errorText}>{error}</p>}
    </section>
  )
}
