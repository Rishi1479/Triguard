import type { Label } from '../api/types'
import { LABEL_STYLES } from '../lib/reasons'
import { useAnimatedNumber } from '../lib/useAnimatedNumber'

const RADIUS = 80
const ARC_LENGTH = Math.PI * RADIUS

/** Animated half-circle gauge for the 0–100% risk score. */
export default function RiskGauge({ score, label }: { score: number; label: Label }) {
  const shown = useAnimatedNumber(score)
  const offset = ARC_LENGTH * (1 - Math.min(Math.max(shown, 0), 100) / 100)
  return (
    <div className="relative mx-auto w-56" role="meter" aria-label="Scam risk score" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(score)} aria-valuetext={`${Math.round(score)} percent risk, ${label}`}>
      <svg viewBox="0 0 200 115" className="w-full" aria-hidden="true">
        <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#e2e8f0" strokeWidth="16" strokeLinecap="round" />
        <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke={LABEL_STYLES[label].stroke} strokeWidth="16" strokeLinecap="round"
          strokeDasharray={ARC_LENGTH} strokeDashoffset={offset} />
      </svg>
      <div className="absolute inset-x-0 bottom-0 text-center">
        <span className="text-4xl font-bold text-navy-900">{Math.round(shown)}%</span>
        <span className="block text-xs text-slate-500">scam risk</span>
      </div>
    </div>
  )
}
