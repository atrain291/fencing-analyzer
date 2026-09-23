import { useParams, useNavigate } from 'react-router-dom'
import { useEffect, useRef, useState, useCallback } from 'react'
import { getBout, deleteBout, Frame, Keypoint } from '@/api/bouts'
import { Trash2, Maximize2, Minimize2 } from 'lucide-react'

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

const SPEEDS = [0.25, 0.5, 1, 2]

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
  for (const kp of Object.values(pose)) {
    if (kp.confidence < CONFIDENCE_THRESHOLD) continue

    ctx.globalAlpha = kp.confidence
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(kp.x * width, kp.y * height, 4, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.globalAlpha = 1
}

export function samplePose(
  frames: Frame[],
  timestampMs: number,
  role: 'fencer_pose' | 'opponent_pose',
): Record<string, Keypoint> | null {
  if (frames.length === 0) return null
  let lo = 0
  let hi = frames.length - 1
  if (timestampMs < frames[lo].timestamp_ms || timestampMs > frames[hi].timestamp_ms) return null

  while (lo + 1 < hi) {
    const mid = (lo + hi) >>> 1
    if (frames[mid].timestamp_ms <= timestampMs) {
      lo = mid
    } else {
      hi = mid
    }
  }

  const a = frames[lo]
  const b = frames[hi]
  if (timestampMs === a.timestamp_ms) return a[role] || null
  if (timestampMs === b.timestamp_ms) return b[role] || null
  const span = b.timestamp_ms - a.timestamp_ms
  if (span <= 0 || span > 100 || !a[role] || !b[role]) return null
  const t = (timestampMs - a.timestamp_ms) / span
  const result: Record<string, Keypoint> = {}
  for (const name of Object.keys(a[role])) {
    const ka = a[role][name]
    const kb = b[role][name]
    if (!kb) continue
    result[name] = {
      x: ka.x + (kb.x - ka.x) * t,
      y: ka.y + (kb.y - ka.y) * t,
      z: ka.z + (kb.z - ka.z) * t,
      confidence: Math.min(ka.confidence, kb.confidence),
    }
  }
  return result
}

export default function VideoReview() {
  const { boutId } = useParams<{ boutId: string }>()
  const navigate = useNavigate()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [analysis, setAnalysis] = useState<AnalysisSummary | null>(null)
  const [videoUrl, setVideoUrl] = useState('')
  const [frames, setFrames] = useState<Frame[]>([])
  const [speed, setSpeed] = useState(1)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const framesRef = useRef<Frame[]>([])

  function handleSpeed(s: number) {
    setSpeed(s)
    if (videoRef.current) videoRef.current.playbackRate = s
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen()
    } else {
      document.exitFullscreen()
    }
  }

  useEffect(() => {
    const onFsChange = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', onFsChange)
    return () => document.removeEventListener('fullscreenchange', onFsChange)
  }, [])

  const renderSkeleton = useCallback((timestampMs?: number) => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    if (!video || !video.getAttribute('src') || !video.videoWidth || !video.videoHeight) {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      canvas.style.width = '0px'
      canvas.style.height = '0px'
      return
    }

    const videoRect = video.getBoundingClientRect()
    const containerRect = containerRef.current?.getBoundingClientRect()
    if (!containerRect || !videoRect.width || !videoRect.height) {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      return
    }
    const scale = Math.min(videoRect.width / video.videoWidth, videoRect.height / video.videoHeight)
    const displayWidth = video.videoWidth * scale
    const displayHeight = video.videoHeight * scale
    canvas.style.left = `${videoRect.left - containerRect.left + (videoRect.width - displayWidth) / 2}px`
    canvas.style.top = `${videoRect.top - containerRect.top + (videoRect.height - displayHeight) / 2}px`
    canvas.style.width = `${displayWidth}px`
    canvas.style.height = `${displayHeight}px`
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    const timeMs = timestampMs ?? video.currentTime * 1000
    const fencerPose = samplePose(framesRef.current, timeMs, 'fencer_pose')
    const opponentPose = samplePose(framesRef.current, timeMs, 'opponent_pose')
    if (fencerPose) drawSkeleton(ctx, fencerPose, canvas.width, canvas.height, '#f97316')
    if (opponentPose) drawSkeleton(ctx, opponentPose, canvas.width, canvas.height, '#3b82f6')
  }, [])

  useEffect(() => {
    let active = true
    framesRef.current = []
    setVideoUrl('')
    setFrames([])
    setAnalysis(null)
    renderSkeleton()
    if (boutId) {
      getBout(Number(boutId)).then((data) => {
        if (!active) return
        const response = data as typeof data & { analysis?: AnalysisSummary | null }
        setVideoUrl(response.video_url ?? '')
        setFrames(Array.isArray(response.frames) ? response.frames : [])
        setAnalysis(response.analysis ?? null)
      })
    }
    return () => { active = false }
  }, [boutId, renderSkeleton])

  useEffect(() => {
    framesRef.current = frames
    renderSkeleton()
  }, [frames, renderSkeleton])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const ro = new ResizeObserver(() => renderSkeleton())
    ro.observe(container)
    if (videoRef.current) ro.observe(videoRef.current)
    return () => ro.disconnect()
  }, [renderSkeleton])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    let running = false
    let frameCallbackId: number | null = null
    let rafId: number | null = null

    const schedule = () => {
      if (!running) return
      if (typeof video.requestVideoFrameCallback === 'function') {
        frameCallbackId = video.requestVideoFrameCallback((_now, metadata) => {
          frameCallbackId = null
          if (!running) return
          renderSkeleton(metadata.mediaTime * 1000)
          schedule()
        })
      } else {
        rafId = requestAnimationFrame(() => {
          rafId = null
          if (!running) return
          renderSkeleton()
          schedule()
        })
      }
    }

    const startLoop = () => {
      if (running) return
      running = true
      renderSkeleton()
      schedule()
    }

    const stopLoop = () => {
      running = false
      if (frameCallbackId !== null) video.cancelVideoFrameCallback(frameCallbackId)
      if (rafId !== null) cancelAnimationFrame(rafId)
      frameCallbackId = null
      rafId = null
    }

    const handlePlay = () => startLoop()
    const handlePause = () => stopLoop()
    const handleEnded = () => stopLoop()
    const handleSeeked = () => renderSkeleton()
    const handleLoadedMetadata = () => renderSkeleton()

    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)
    video.addEventListener('ended', handleEnded)
    video.addEventListener('seeked', handleSeeked)
    video.addEventListener('loadedmetadata', handleLoadedMetadata)

    if (!video.paused && !video.ended) startLoop()

    return () => {
      stopLoop()
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
      video.removeEventListener('ended', handleEnded)
      video.removeEventListener('seeked', handleSeeked)
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
    }
  }, [renderSkeleton])

  return (
    <div className="max-w-screen-2xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold">Bout Review — #{boutId}</h1>
        <button
          onClick={async () => {
            if (!window.confirm('Delete this bout and its video? This cannot be undone.')) return
            await deleteBout(Number(boutId))
            navigate('/')
          }}
          className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-500 rounded-lg text-sm font-medium transition-colors"
        >
          <Trash2 size={14} /> Delete Bout
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Video + skeleton canvas */}
        <div className="lg:col-span-3 space-y-3">
          <div
            ref={containerRef}
            className={`relative bg-black rounded-xl overflow-hidden aspect-video ${isFullscreen ? 'rounded-none w-screen h-screen' : ''}`}
          >
            <button
              onClick={toggleFullscreen}
              className="absolute top-2 right-2 z-10 p-1.5 bg-black/50 hover:bg-black/70 rounded-md text-white transition-colors"
              title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen with overlay'}
            >
              {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            </button>
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              className="w-full h-full object-contain"
            />
            <canvas
              ref={canvasRef}
              className="absolute pointer-events-none"
            />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Speed</span>
            {SPEEDS.map(s => (
              <button
                key={s}
                onClick={() => handleSpeed(s)}
                className={[
                  'px-2.5 py-1 rounded-md text-xs font-medium transition-colors',
                  speed === s
                    ? 'bg-brand-500 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700',
                ].join(' ')}
              >
                {s}×
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-500">
            {frames.length > 0
              ? `Skeleton overlay active — ${frames.length} frames loaded`
              : 'No pose data available'}
          </p>
        </div>

        {/* Analysis panel */}
        <div className="lg:col-span-1 space-y-4">
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
