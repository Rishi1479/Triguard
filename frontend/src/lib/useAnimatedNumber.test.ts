import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useAnimatedNumber } from './useAnimatedNumber'

afterEach(() => vi.restoreAllMocks())

describe('useAnimatedNumber', () => {
  it('jumps straight to the target when reduced motion is preferred', () => {
    vi.stubGlobal('matchMedia', () => ({ matches: true }) as MediaQueryList)
    const { result } = renderHook(() => useAnimatedNumber(87))
    expect(result.current).toBe(87)
  })

  it('never returns a negative value when the frame timestamp precedes the start', () => {
    // Headless/coalesced frames can hand rAF a timestamp older than performance.now().
    vi.stubGlobal('matchMedia', () => ({ matches: false }) as MediaQueryList)
    vi.spyOn(performance, 'now').mockReturnValue(10_000)
    let frame: FrameRequestCallback | null = null
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => { frame = cb; return 1 })
    vi.stubGlobal('cancelAnimationFrame', () => {})

    const { result } = renderHook(() => useAnimatedNumber(87))
    act(() => { frame?.(9_000) }) // 1s *before* the recorded start
    expect(result.current).toBeGreaterThanOrEqual(0)
  })

  it('reaches the target once the duration has elapsed', () => {
    vi.stubGlobal('matchMedia', () => ({ matches: false }) as MediaQueryList)
    vi.spyOn(performance, 'now').mockReturnValue(0)
    let frame: FrameRequestCallback | null = null
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => { frame = cb; return 1 })
    vi.stubGlobal('cancelAnimationFrame', () => {})

    const { result } = renderHook(() => useAnimatedNumber(87, 900))
    act(() => { frame?.(900) })
    expect(result.current).toBeCloseTo(87)
  })
})
