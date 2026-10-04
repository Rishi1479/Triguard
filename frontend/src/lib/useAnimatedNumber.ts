import { useEffect, useState } from 'react'

const prefersReducedMotion = () =>
  typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

/** Eases a number from 0 to `target` over `durationMs` (jumps instantly with reduced motion or in tests). */
export function useAnimatedNumber(target: number, durationMs = 900): number {
  const [value, setValue] = useState(0)
  useEffect(() => {
    if (prefersReducedMotion() || typeof requestAnimationFrame !== 'function') {
      setValue(target)
      return
    }
    let raf = 0
    const start = performance.now()
    const tick = (now: number) => {
      // Clamp low as well as high: a rAF timestamp can precede the captured start (different
      // time origins), which would otherwise ease through negative values.
      const t = Math.min(Math.max((now - start) / durationMs, 0), 1)
      setValue(target * (1 - Math.pow(1 - t, 3)))
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, durationMs])
  return value
}
