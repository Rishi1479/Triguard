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
export const ChevronIcon = (p: P) => (
  <svg {...base} width={16} height={16} {...p}><path d="M9 6l6 6-6 6" /></svg>
)
export const TextIcon = (p: P) => (
  <svg {...base} {...p}><path d="M4 6h16M4 12h12M4 18h8" /></svg>
)
export const ImageIcon = (p: P) => (
  <svg {...base} {...p}><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="8.5" cy="9.5" r="1.5" /><path d="M21 16l-5-5-4 4-2-2-4 4" /></svg>
)
export const LinkIcon = (p: P) => (
  <svg {...base} {...p}><path d="M10 13a5 5 0 007 0l2-2a5 5 0 00-7-7l-1 1" /><path d="M14 11a5 5 0 00-7 0l-2 2a5 5 0 007 7l1-1" /></svg>
)
export const SunIcon = (p: P) => (
  <svg {...base} {...p}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19" /></svg>
)
export const MoonIcon = (p: P) => (
  <svg {...base} {...p}><path d="M20 14.5A8 8 0 019.5 4a8 8 0 1010.5 10.5z" /></svg>
)
export const MonitorIcon = (p: P) => (
  <svg {...base} {...p}><rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" /></svg>
)
export const ChartIcon = (p: P) => (
  <svg {...base} {...p}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></svg>
)
export const SparkIcon = (p: P) => (
  <svg {...base} {...p}><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z" /></svg>
)
export const SearchIcon = (p: P) => (
  <svg {...base} {...p}><circle cx="11" cy="11" r="7" /><path d="M20 20l-4.5-4.5" /></svg>
)
export const CopyIcon = (p: P) => (
  <svg {...base} {...p}><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V5a2 2 0 012-2h8" /></svg>
)

/** Severity icon + colour. Paired with text everywhere so colour is never the only cue. */
export function SeverityIcon({ severity }: { severity: Severity }) {
  const style = {
    high: { color: 'var(--risk-scam)' },
    medium: { color: 'var(--risk-suspicious)' },
    low: { color: 'var(--ink-muted)' },
  }[severity]
  if (severity === 'high') return <XCircleIcon className="shrink-0" style={style} />
  if (severity === 'medium') return <AlertIcon className="shrink-0" style={style} />
  return <InfoIcon className="shrink-0" style={style} />
}

export const MODALITY_ICON = { text: TextIcon, image: ImageIcon, url: LinkIcon } as const

/** Verdict icon per label — colour is never the only cue, so every use pairs this with the name. */
export const RISK_ICON = {
  Safe: ShieldIcon,
  Suspicious: AlertIcon,
  'Likely Scam': XCircleIcon,
} as const
