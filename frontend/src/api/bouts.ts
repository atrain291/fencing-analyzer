import api from './client'

export interface Keypoint {
  x: number
  y: number
  z: number
  confidence: number
}

export interface BladeState {
  tip_xyz: { x: number; y: number; z: number }
  nominal_xyz: { x: number; y: number; z: number } | null
  velocity_xyz: { x: number; y: number; z: number }
  speed: number | null
  confidence: number | null
}

export interface MeshState {
  subject: string
  joints_3d: Record<string, { x: number; y: number; z: number }>
  global_translation: { x: number; y: number; z: number } | null
  foot_contact: { left_heel: number; left_toe: number; right_heel: number; right_toe: number } | null
  confidence: number | null
}

export interface Frame {
  id: number
  timestamp_ms: number
  fencer_pose: Record<string, Keypoint>
  opponent_pose: Record<string, Keypoint> | null
  blade_state: BladeState | null
  opponent_blade_state: BladeState | null
  mesh_states: MeshState[]
}

export interface PipelineProgress {
  stage?: string
  pct?: number
  frame?: number
  total_frames?: number
  gpu_mem_pct?: number
  cpu_pct?: number
}

export interface Action {
  id: number
  bout_id: number
  subject: string
  type: string
  start_ms: number
  end_ms: number
  outcome: string | null
  confidence: number | null
}

export interface Analysis {
  id: number
  bout_id: number
  coaching_text: string | null
  patterns: Record<string, any> | null
  practice_plan: Record<string, any> | null
}

export interface Bout {
  id: number
  session_id: number
  status: string
  result: string | null
  video_url: string | null
  duration_ms: number | null
  pipeline_progress: PipelineProgress
  created_at: string
  analysis: Analysis | null
}

export interface BoutFramesResponse {
  frames: Frame[]
  actions: Action[]
}

export interface BoutStatus {
  bout_id: number
  status: string
  pipeline_progress: PipelineProgress
  error?: string
}

export interface BoutUploadResponse {
  bout_id: number
  task_id: string
  status: string
}

export async function uploadVideo(
  file: File,
  fencerId: number,
  onProgress?: (pct: number) => void,
  signal?: AbortSignal,
): Promise<BoutUploadResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('fencer_id', String(fencerId))

  const { data } = await api.post<BoutUploadResponse>('/upload/', form, {
    timeout: 0,
    onUploadProgress: e => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
    signal,
  })
  return data
}

export async function getBoutStatus(boutId: number): Promise<BoutStatus> {
  const { data } = await api.get<BoutStatus>(`/bouts/${boutId}/status`)
  return data
}

export async function getBout(boutId: number): Promise<Bout> {
  const { data } = await api.get<Bout>(`/bouts/${boutId}`)
  return data
}

export async function getBoutFrames(boutId: number): Promise<BoutFramesResponse> {
  const { data } = await api.get<BoutFramesResponse>(`/bouts/${boutId}/frames`)
  return data
}

export async function deleteBout(boutId: number): Promise<void> {
  await api.delete(`/bouts/${boutId}`)
}

export interface Bbox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface DetectionSelection {
  frame_index: number
  detection_index: number
}

export interface PreviewDetection {
  index: number
  bbox: Bbox
  confidence: number
  keypoints: Record<string, Keypoint>
}

export interface PreviewFrame {
  frame_index: number
  timestamp_ms: number
  image_key: string
  detections: PreviewDetection[]
}

export interface PreviewResponse {
  status: 'processing' | 'ready' | 'failed'
  preview_data?: {
    frames: PreviewFrame[]
  }
  error?: string
}

export interface BoutSummary {
  id: number
  status: string
  created_at: string
  video_url: string | null
  duration_ms: number | null
}

export function getThumbnailUrl(boutId: number): string {
  return `${api.defaults.baseURL}/bouts/${boutId}/thumbnail`
}

export interface ActionBreakdownEntry {
  count: number
  avg_duration_ms: number
  consistency_score: number
}

export interface DrillReport {
  drill_type: string
  total_actions: number
  total_duration_ms: number
  tempo: number
  action_breakdown: Record<string, ActionBreakdownEntry>
  rhythm_score: number
  tempo_score: number
  overall_score: number
}

export async function getDrillReport(boutId: number): Promise<DrillReport> {
  const { data } = await api.get<DrillReport>(`/bouts/${boutId}/drill-report`)
  return data
}

export async function configureROI(
  boutId: number,
  fencerBbox: Bbox | null,
  opponentBbox: Bbox | null,
  fencerDetection?: DetectionSelection | null,
  opponentDetection?: DetectionSelection | null,
): Promise<{ status: string; task_id: string }> {
  const { data } = await api.post(`/bouts/${boutId}/roi`, {
    fencer_bbox: fencerBbox,
    opponent_bbox: opponentBbox,
    fencer_detection: fencerDetection ?? undefined,
    opponent_detection: opponentDetection ?? undefined,
  })
  return data
}

export async function getPreview(boutId: number): Promise<PreviewResponse> {
  const { data } = await api.get<PreviewResponse>(`/bouts/${boutId}/preview`)
  return data
}

export async function triggerPreview(boutId: number): Promise<{ status: string; task_id: string }> {
  const { data } = await api.post(`/bouts/${boutId}/preview`)
  return data
}

export async function listBouts(fencerId: number): Promise<BoutSummary[]> {
  const { data } = await api.get<BoutSummary[]>('/bouts/', { params: { fencer_id: fencerId } })
  return data
}
