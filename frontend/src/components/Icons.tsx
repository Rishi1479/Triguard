import type { SVGProps } from 'react'
import type { Severity } from '../api/types'

type P = SVGProps<SVGSVGElement>
const base = { width: 20, height: 20, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true }

export const ShieldIcon = (p: P) => (
  <svg {...base} {...p}><path d="M12 3l8 3v6c0 4.5-3.2 8.2-8 9-4.8-.8-8-4.5-8-9V6l8-3z" /><path d="M9 12l2 2 4-4" /></svg>
)
export const AlertIcon = (p: P) => (
  <svg {...base} {...p}><path d="M12 3l10 18H2L12 3z" /><path d="M12 10v5M12 18h.01" /></svg>
)
export const XCircleIcon = (p: P) => (
  <svg {...base} {...p}><circle cx="12" cy="12" r="9" /><path d="M15 9l-6 6M9 9l6 6" /></svg>
)
export const InfoIcon = (p: P) => (
  <svg {...base} {...p}><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></svg>
)
export const UploadIcon = (p: P) => (
  <svg {...base} {...p}><path d="M12 16V4M7 9l5-5 5 5" /><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" /></svg>
)

/** Severity icon + sr-only text so colour is never the only cue. */
export function SeverityIcon({ severity }: { severity: Severity }) {
  if (severity === 'high') return <XCircleIcon className="shrink-0 text-red-600" />
  if (severity === 'medium') return <AlertIcon className="shrink-0 text-amber-600" />
  return <InfoIcon className="shrink-0 text-slate-500" />
}
