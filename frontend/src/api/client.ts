import type { AnalyzeInput, AnalyzeResponse, HistoryItem, Stats } from './types'

const BASE = import.meta.env.VITE_API_BASE ?? ''
export const ANALYZE_TIMEOUT_MS = 120_000
export const DEFAULT_TIMEOUT_MS = 15_000

/** Error with a machine-readable code (backend codes plus `timeout`, `network`, `cancelled`). */
export class ApiError extends Error {
  code: string
  status: number
  requestId?: string

  constructor(code: string, message: string, status = 0, requestId?: string) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.requestId = requestId
  }
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController()
  const outer = init.signal
  if (outer) outer.addEventListener('abort', () => controller.abort('cancelled'), { once: true })
  const timer = setTimeout(() => controller.abort('timeout'), timeoutMs)
  try {
    const res = await fetch(`${BASE}${path}`, { ...init, signal: controller.signal })
    if (!res.ok) {
      let body: { error?: { code?: string; message?: string; request_id?: string } } = {}
      try {
        body = await res.json()
      } catch {
        /* non-JSON error body */
      }
      throw new ApiError(
        body.error?.code ?? `http_${res.status}`,
        body.error?.message ?? `Request failed (${res.status}).`,
        res.status,
        body.error?.request_id,
      )
    }
    return (await res.json()) as T
  } catch (err) {
    if (err instanceof ApiError) throw err
    if (controller.signal.aborted) {
      if (controller.signal.reason === 'cancelled') throw new ApiError('cancelled', 'Request cancelled.')
      throw new ApiError('timeout', 'The server took too long to respond. Please try again.')
    }
    throw new ApiError('network', 'Could not reach the TriGuard server. Check your connection and try again.')
  } finally {
    clearTimeout(timer)
  }
}

/** POST /api/analyze with any combination of text, url and image. */
export function analyze(input: AnalyzeInput, signal?: AbortSignal): Promise<AnalyzeResponse> {
  const form = new FormData()
  if (input.text?.trim()) form.append('text', input.text.trim())
  if (input.url?.trim()) form.append('url', input.url.trim())
  if (input.image) form.append('image', input.image)
  return request<AnalyzeResponse>('/api/analyze', { method: 'POST', body: form, signal }, ANALYZE_TIMEOUT_MS)
}

/** POST /api/feedback ("Was this correct?"). */
export function sendFeedback(analysisId: number, correct: boolean, comment?: string) {
  return request<{ id: number; status: string }>('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ analysis_id: analysisId, correct, comment: comment || null }),
  })
}

export const getHistory = (signal?: AbortSignal) => request<HistoryItem[]>('/api/history', { signal })
export const getStats = (signal?: AbortSignal) => request<Stats>('/api/stats', { signal })
