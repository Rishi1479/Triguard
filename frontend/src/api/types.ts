/** Types mirroring the backend (ml/common/schemas.py + backend/app/routes.py). */
export type Severity = 'low' | 'medium' | 'high'
export type ModalityName = 'text' | 'image' | 'url'
export type Label = 'Safe' | 'Suspicious' | 'Likely Scam'
/** 'low' when the backend abstained: the verdict is provisional. */
export type EvidenceStrength = 'low' | 'normal'

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
  /** True when the evidence does not justify a firm verdict; `label` is then provisional. */
  abstained?: boolean
  evidence_strength?: EvidenceStrength
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

export interface HistoryDetail {
  id: number
  created_at: string
  inputs_present: string
  submitted_text: string | null
  url_domain: string | null
  image_submitted: boolean
  latency_ms: number
  detail_available: boolean
  feedback: { correct: boolean; comment: string | null; created_at: string }[]
  result: AnalyzeResponse
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
