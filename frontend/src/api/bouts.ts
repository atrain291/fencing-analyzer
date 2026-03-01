import api from './client'

export interface Keypoint {
  x: number
  y: number
  z: number
  confidence: number
}

export interface Frame {
  id: number
  timestamp_ms: number
  fencer_pose: Record<string, Keypoint>
  opponent_pose: Record<string, Keypoint> | null
}

export interface Bout {
  id: number
  session_id: number
  status: string
  result: string | null
  video_url: string | null
  duration_ms: number | null
  pipeline_progress: { stage?: string; pct?: number }
  created_at: string
  frames: Frame[]
}

export interface BoutStatus {
  bout_id: number
  status: string
  pipeline_progress: { stage?: string; pct?: number }
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
): Promise<BoutUploadResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('fencer_id', String(fencerId))

  const { data } = await api.post<BoutUploadResponse>('/upload/', form, {
    onUploadProgress: e => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
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
