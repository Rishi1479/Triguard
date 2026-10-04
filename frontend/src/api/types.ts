/** Types mirroring the backend (ml/common/schemas.py + backend/app/routes.py). */
export type Severity = 'low' | 'medium' | 'high'
export type ModalityName = 'text' | 'image' | 'url'
export type Label = 'Safe' | 'Suspicious' | 'Likely Scam'

export interface RedFlag {
  code: string
  message: string
  severity: Severity
  evidence: string
}

export interface ModalityResult {
  modality: ModalityName
  probability: number
  red_flags: RedFlag[]
  extracted: Record<string, unknown>
  explanation: Record<string, unknown>
  available: boolean
}

export interface FinalResult {
  label: Label
  risk_score: number
  modality_results: ModalityResult[]
  reasons: string[]
  cross_modal_flags: RedFlag[]
}

export interface AnalyzeResponse extends FinalResult {
  id: number
  latency_ms: number
}

export interface HistoryItem {
  id: number
  created_at: string
  inputs_present: string
  label: Label
  risk_score: number
  url_domain: string | null
  text_preview: string | null
  reasons: string[]
  latency_ms: number
}

export interface Stats {
  total_analyses: number
  label_counts: Partial<Record<Label, number>>
  input_combinations: Record<string, number>
  avg_risk_score: number | null
  avg_latency_ms: number | null
  feedback: { total: number; marked_correct: number; accuracy: number | null }
}

export interface AnalyzeInput {
  text?: string
  url?: string
  image?: File | null
}
