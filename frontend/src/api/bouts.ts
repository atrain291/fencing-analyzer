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
}

export interface Frame {
  id: number
  timestamp_ms: number
  fencer_pose: Record<string, Keypoint>
  opponent_pose: Record<string, Keypoint> | null
  blade_state: BladeState | null
}

export interface PipelineProgress {
  stage?: string
  pct?: number
  frame?: number
  total_frames?: number
  gpu_mem_pct?: number
  cpu_pct?: number
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
  frames: Frame[]
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

export async function deleteBout(boutId: number): Promise<void> {
  await api.delete(`/bouts/${boutId}`)
}

export interface Bbox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export function getThumbnailUrl(boutId: number): string {
  return `${api.defaults.baseURL}/bouts/${boutId}/thumbnail`
}

export async function configureROI(
  boutId: number,
  fencerBbox: Bbox | null,
  opponentBbox: Bbox | null,
): Promise<{ status: string; task_id: string }> {
  const { data } = await api.post(`/bouts/${boutId}/roi`, {
    fencer_bbox: fencerBbox,
    opponent_bbox: opponentBbox,
  })
  return data
}
