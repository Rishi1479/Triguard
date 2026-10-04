import { useEffect, useState } from 'react'

const STEP_MS = 1600

/** Cycles through plain-language progress messages for the inputs actually provided. */
export function stepsFor(has: { text: boolean; image: boolean; url: boolean }): string[] {
  const steps: string[] = []
  if (has.text) steps.push('Reading the text for scam wording…')
  if (has.image) steps.push('Reading the image with OCR and looking for QR codes…')
  if (has.url) steps.push('Checking the link for look-alike domains (without opening it)…')
  if ([has.text, has.image, has.url].filter(Boolean).length > 1) steps.push('Cross-checking that everything matches…')
  steps.push('Combining the evidence into a risk score…')
  return steps
}

export default function LoadingSteps({ steps, onCancel }: { steps: string[]; onCancel: () => void }) {
  const [i, setI] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setI((n) => Math.min(n + 1, steps.length - 1)), STEP_MS)
    return () => clearInterval(t)
  }, [steps.length])
  return (
    <div
      className="rounded-[var(--radius-card)] border p-6 text-center"
      style={{ borderColor: 'var(--line-hairline)', background: 'var(--surface-card)' }}
      role="status"
      aria-live="polite"
    >
      <div
        className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-4"
        style={{ borderColor: 'var(--line-hairline)', borderTopColor: 'var(--accent)' }}
        aria-hidden="true"
      />
      <p className="font-semibold text-[var(--ink-primary)]">{steps[i]}</p>
      <ol className="mx-auto mt-3 max-w-md space-y-1 text-left text-sm text-[var(--ink-muted)]">
        {steps.map((s, idx) => (
          <li
            key={s}
            className={idx === i ? 'font-semibold text-[var(--ink-primary)]' : ''}
            style={idx < i ? { color: 'var(--risk-safe-ink)' } : undefined}
          >
            {idx < i ? '✓ ' : idx === i ? '• ' : '○ '}
            {s}
          </li>
        ))}
      </ol>
      <button type="button" onClick={onCancel} className="mt-4 text-sm text-[var(--ink-secondary)] underline">Cancel</button>
    </div>
  )
}
