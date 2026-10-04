import type { Label } from '../api/types'
import { RISK_TOKENS } from '../lib/riskTokens'
import { useAnimatedNumber } from '../lib/useAnimatedNumber'

const RADIUS = 80
const ARC_LENGTH = Math.PI * RADIUS
const ARC = 'M20 100 A80 80 0 0 1 180 100'

/**
 * Animated half-circle gauge for the 0–100% risk score. The number is the primary read; the
 * decision bands live in RiskBandMeter beside it.
 */
export default function RiskGauge({ score, label }: { score: number; label: Label }) {
  const shown = useAnimatedNumber(score)
  const offset = ARC_LENGTH * (1 - Math.min(Math.max(shown, 0), 100) / 100)
  return (
    <div
      className="relative mx-auto w-48 sm:w-56"
      role="meter"
      aria-label="Scam risk score"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(score)}
      aria-valuetext={`${Math.round(score)} percent risk, ${label}`}
    >
      <svg viewBox="0 0 200 115" className="w-full" aria-hidden="true">
        <path d={ARC} fill="none" stroke="var(--viz-track)" strokeWidth="14" strokeLinecap="round" />
        {shown > 0.5 && (
          <path
            d={ARC}
            fill="none"
            stroke={RISK_TOKENS[label].bar}
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={ARC_LENGTH}
            strokeDashoffset={offset}
          />
        )}
      </svg>
      <div className="absolute inset-x-0 bottom-0 text-center">
        <span className="text-4xl font-bold text-[var(--ink-primary)]">{Math.round(shown)}%</span>
        <span className="block text-xs text-[var(--ink-muted)]">scam risk</span>
      </div>
    </div>
  )
}
