/** Shared UI primitives. Every colour comes from a token in index.css. */
import type { ReactNode } from 'react'
import { useId, useState } from 'react'
import { ChevronIcon } from './Icons'

type Div = React.HTMLAttributes<HTMLDivElement>

export function Card({ children, className = '', ...rest }: Div & { children: ReactNode }) {
  return (
    <div
      {...rest}
      className={`rounded-[var(--radius-card)] border bg-[var(--surface-card)] ${className}`}
      style={{ borderColor: 'var(--line-hairline)', ...rest.style }}
    >
      {children}
    </div>
  )
}

export function SectionTitle({ children, id, hint }: { children: ReactNode; id?: string; hint?: ReactNode }) {
  return (
    <div className="mb-3">
      <h2 id={id} className="text-lg font-bold text-[var(--ink-primary)]">{children}</h2>
      {hint && <p className="mt-0.5 text-sm text-[var(--ink-secondary)]">{hint}</p>}
    </div>
  )
}

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
}

const SIZES = { sm: 'px-3 py-1.5 text-sm', md: 'px-4 py-2 text-sm', lg: 'px-6 py-3 text-base' }

export function Button({ variant = 'primary', size = 'md', className = '', ...rest }: ButtonProps) {
  const base = `inline-flex items-center justify-center gap-2 rounded-lg font-semibold transition-colors
    disabled:cursor-not-allowed ${SIZES[size]}`
  const styles: Record<string, string> = {
    // A disabled primary must not still read as the call to action.
    primary:
      'bg-[var(--accent)] text-[var(--accent-ink)] hover:bg-[var(--accent-hover)] disabled:bg-[var(--surface-sunken)] disabled:text-[var(--ink-muted)]',
    secondary:
      'border bg-[var(--surface-card)] text-[var(--ink-primary)] hover:bg-[var(--surface-sunken)] border-[var(--line-strong)] disabled:opacity-60',
    ghost: 'text-[var(--ink-secondary)] hover:bg-[var(--surface-sunken)] disabled:opacity-60',
  }
  return <button {...rest} className={`${base} ${styles[variant]} ${className}`} />
}

/** Small labelled chip. `icon` is required for risk/status use so colour is never alone. */
export function Chip({ children, tone, icon, className = '' }: {
  children: ReactNode
  tone: { wash: string; ink: string }
  icon?: ReactNode
  className?: string
}) {
  return (
    <span
      className={`risk-chip inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${className}`}
      style={{ background: tone.wash, color: tone.ink }}
    >
      {icon}
      {children}
    </span>
  )
}

/** Key figure. Proportional figures by default; pass `tabular` for aligned columns. */
export function StatTile({ label, value, unit, hint, tabular = false }: {
  label: string
  value: string | number
  unit?: string
  hint?: string
  tabular?: boolean
}) {
  return (
    <Card className="p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-[var(--ink-muted)]">{label}</p>
      <p className={`mt-1 text-3xl font-bold leading-none text-[var(--ink-primary)] ${tabular ? 'tabular' : ''}`}>
        {value}
        {unit && <span className="ml-1 text-base font-semibold text-[var(--ink-secondary)]">{unit}</span>}
      </p>
      {hint && <p className="mt-1.5 text-xs text-[var(--ink-secondary)]">{hint}</p>}
    </Card>
  )
}

export function Accordion({ summary, children, defaultOpen = false }: {
  summary: ReactNode
  children: ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  const id = useId()
  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-start gap-2 rounded-md px-1 py-1 text-left hover:bg-[var(--surface-sunken)]"
      >
        <ChevronIcon className={`mt-0.5 shrink-0 transition-transform ${open ? 'rotate-90' : ''}`} />
        <span className="flex-1">{summary}</span>
      </button>
      {open && <div id={id} className="pl-7 pr-1 pb-2 text-sm text-[var(--ink-secondary)]">{children}</div>}
    </div>
  )
}

export function Skeleton({ className = '' }: { className?: string }) {
  return <div aria-hidden="true" className={`animate-pulse rounded-md bg-[var(--surface-sunken)] ${className}`} />
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <Card className="p-8 text-center">
      <p className="font-semibold text-[var(--ink-primary)]">{title}</p>
      {children && <p className="mx-auto mt-1 max-w-md text-sm text-[var(--ink-secondary)]">{children}</p>}
    </Card>
  )
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <p
      role="alert"
      className="rounded-lg border p-3 text-sm font-medium"
      style={{
        background: 'var(--risk-scam-wash)',
        color: 'var(--risk-scam-ink)',
        borderColor: 'var(--risk-scam)',
      }}
    >
      {children}
    </p>
  )
}
