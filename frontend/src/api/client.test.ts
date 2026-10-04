import { afterEach, describe, expect, it, vi } from 'vitest'
import { analyze, ApiError, sendFeedback } from './client'
import { scamResult } from '../test/fixtures'

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers() })

describe('api client', () => {
  it('posts multipart form data with only provided fields', async () => {
    const spy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(json(scamResult))
    const res = await analyze({ text: ' hello ', url: '', image: new File(['x'], 'a.png', { type: 'image/png' }) })
    expect(res.label).toBe('Likely Scam')
    const [url, init] = spy.mock.calls[0]
    expect(url).toBe('/api/analyze')
    const form = (init as RequestInit).body as FormData
    expect(form.get('text')).toBe('hello')
    expect(form.has('url')).toBe(false)
    expect((form.get('image') as File).name).toBe('a.png')
  })

  it('turns backend error bodies into ApiError', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(json({ error: { code: 'no_input', message: 'Provide input', request_id: 'r1' } }, 400))
    await expect(analyze({})).rejects.toMatchObject({ code: 'no_input', message: 'Provide input', status: 400, requestId: 'r1' })
  })

  it('handles non-JSON error bodies and network failures', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(new Response('<html>bad gateway</html>', { status: 502 }))
    await expect(analyze({ text: 'x' })).rejects.toMatchObject({ code: 'http_502', status: 502 })
    vi.spyOn(globalThis, 'fetch').mockRejectedValueOnce(new TypeError('Failed to fetch'))
    await expect(analyze({ text: 'x' })).rejects.toMatchObject({ code: 'network' })
  })

  it('times out and supports cancellation', async () => {
    vi.useFakeTimers()
    vi.spyOn(globalThis, 'fetch').mockImplementation((_u, init) => new Promise((_r, rej) => {
      (init as RequestInit).signal!.addEventListener('abort', () => rej(new DOMException('aborted', 'AbortError')))
    }))
    const p = sendFeedback(1, true)
    const assertion = expect(p).rejects.toMatchObject({ code: 'timeout' })
    await vi.advanceTimersByTimeAsync(16_000)
    await assertion
    const c = new AbortController()
    const q = analyze({ text: 'x' }, c.signal)
    const cancelled = expect(q).rejects.toBeInstanceOf(ApiError)
    c.abort()
    await cancelled
  })

  it('sends feedback as JSON', async () => {
    const spy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(json({ id: 1, status: 'recorded' }, 201))
    await sendFeedback(7, false)
    expect(JSON.parse((spy.mock.calls[0][1] as RequestInit).body as string)).toEqual({ analysis_id: 7, correct: false, comment: null })
  })
})
