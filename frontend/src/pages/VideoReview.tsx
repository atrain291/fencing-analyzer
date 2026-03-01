import { useParams } from 'react-router-dom'
import { useEffect, useRef, useState, useCallback } from 'react'
import { getBout, Frame, Keypoint } from '@/api/bouts'

interface AnalysisSummary {
  llm_summary: string
  technique_scores: Record<string, number>
}

const SKELETON_EDGES: [string, string][] = [
  ['nose', 'left_eye'], ['nose', 'right_eye'],
  ['left_eye', 'left_ear'], ['right_eye', 'right_ear'],
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
]

const CONFIDENCE_THRESHOLD = 0.3

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  pose: Record<string, Keypoint>,
  width: number,
  height: number,
  color: string
) {
  // Draw edges
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  for (const [start, end] of SKELETON_EDGES) {
    const p1 = pose[start]
    const p2 = pose[end]
    if (!p1 || !p2) continue
    if (p1.confidence < CONFIDENCE_THRESHOLD || p2.confidence < CONFIDENCE_THRESHOLD) continue

    const alpha = Math.min(p1.confidence, p2.confidence)
    ctx.globalAlpha = alpha
    ctx.beginPath()
    ctx.moveTo(p1.x * width, p1.y * height)
    ctx.lineTo(p2.x * width, p2.y * height)
    ctx.stroke()
  }

  // Draw keypoints
  ctx.globalAlpha = 1
  for (const [name, kp] of Object.entries(pose)) {
    if (kp.confidence < CONFIDENCE_THRESHOLD) continue

    ctx.globalAlpha = kp.confidence
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(kp.x * width, kp.y * height, 4, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.globalAlpha = 1
}

function findClosestFrame(frames: Frame[], timestampMs: number): Frame | null {
  if (frames.length === 0) return null

  let closest = frames[0]
  let minDiff = Math.abs(frames[0].timestamp_ms - timestampMs)

  for (const frame of frames) {
    const diff = Math.abs(frame.timestamp_ms - timestampMs)
    if (diff < minDiff) {
      minDiff = diff
      closest = frame
    }
  }

  return closest
}

export default function VideoReview() {
  const { boutId } = useParams<{ boutId: string }>()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [analysis, setAnalysis] = useState<AnalysisSummary | null>(null)
  const [videoUrl, setVideoUrl] = useState('')
  const [frames, setFrames] = useState<Frame[]>([])

  useEffect(() => {
    if (!boutId) return
    getBout(Number(boutId)).then((data) => {
      if (data.video_url) setVideoUrl(data.video_url)
      if ((data as any).analysis) setAnalysis((data as any).analysis)
      if (data.frames) setFrames(data.frames)
    })
  }, [boutId])

  const renderSkeleton = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || frames.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Sync canvas size with video
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth || video.clientWidth
      canvas.height = video.videoHeight || video.clientHeight
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Find closest frame to current video time
    const currentTimeMs = video.currentTime * 1000
    const frame = findClosestFrame(frames, currentTimeMs)
    if (!frame) return

    // Draw fencer skeleton (orange)
    if (frame.fencer_pose) {
      drawSkeleton(ctx, frame.fencer_pose, canvas.width, canvas.height, '#f97316')
    }

    // Draw opponent skeleton (blue)
    if (frame.opponent_pose) {
      drawSkeleton(ctx, frame.opponent_pose, canvas.width, canvas.height, '#3b82f6')
    }
  }, [frames])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleTimeUpdate = () => renderSkeleton()
    const handleLoadedMetadata = () => renderSkeleton()
    const handleSeeked = () => renderSkeleton()

    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    video.addEventListener('seeked', handleSeeked)

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      video.removeEventListener('seeked', handleSeeked)
    }
  }, [renderSkeleton])

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <h1 className="text-xl font-bold">Bout Review — #{boutId}</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video + skeleton canvas */}
        <div className="lg:col-span-2 space-y-3">
          <div className="relative bg-black rounded-xl overflow-hidden aspect-video">
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              className="w-full h-full object-contain"
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />
          </div>
          <p className="text-xs text-gray-500">
            {frames.length > 0
              ? `Skeleton overlay active — ${frames.length} frames loaded`
              : 'No pose data available'}
          </p>
        </div>

        {/* Analysis panel */}
        <div className="space-y-4">
          <div className="bg-gray-900 rounded-xl p-4">
            <h2 className="font-semibold mb-3 text-sm text-gray-400 uppercase tracking-wider">
              AI Coaching Feedback
            </h2>
            {analysis ? (
              <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
                {analysis.llm_summary}
              </p>
            ) : (
              <p className="text-sm text-gray-500">Loading analysis...</p>
            )}
          </div>

          <div className="bg-gray-900 rounded-xl p-4">
            <h2 className="font-semibold mb-3 text-sm text-gray-400 uppercase tracking-wider">
              Technique Scores
            </h2>
            <p className="text-sm text-gray-500">
              Detailed scoring available after Stage 2+ pipeline completion.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
