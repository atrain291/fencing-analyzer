import { useParams, useNavigate } from 'react-router-dom'
import { useEffect, useRef, useState, useCallback } from 'react'
import { getBout, deleteBout, Frame, Keypoint, BladeState } from '@/api/bouts'
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

function drawBlade(
  ctx: CanvasRenderingContext2D,
  pose: Record<string, Keypoint>,
  bladeState: BladeState,
  width: number,
  height: number
) {
  const wrist = pose['right_wrist'] ?? pose['left_wrist']
  if (!wrist || wrist.confidence < 0.3) return

  const tip = bladeState.tip_xyz
  ctx.strokeStyle = '#22c55e'
  ctx.lineWidth = 2
  ctx.globalAlpha = 0.85
  ctx.beginPath()
  ctx.moveTo(wrist.x * width, wrist.y * height)
  ctx.lineTo(tip.x * width, tip.y * height)
  ctx.stroke()

  ctx.fillStyle = '#22c55e'
  ctx.globalAlpha = 1.0
  ctx.beginPath()
  ctx.arc(tip.x * width, tip.y * height, 5, 0, Math.PI * 2)
  ctx.fill()
  ctx.globalAlpha = 1
}

function findFrameInterval(
  frames: Frame[],
  timestampMs: number
): { a: Frame; b: Frame; t: number } | null {
  if (frames.length === 0) return null
  if (frames.length === 1) return { a: frames[0], b: frames[0], t: 0 }

  let lo = 0
  let hi = frames.length - 1

  if (timestampMs <= frames[lo].timestamp_ms) return { a: frames[lo], b: frames[lo], t: 0 }
  if (timestampMs >= frames[hi].timestamp_ms) return { a: frames[hi], b: frames[hi], t: 0 }

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
  const span = b.timestamp_ms - a.timestamp_ms
  const t = span > 0 ? (timestampMs - a.timestamp_ms) / span : 0
  return { a, b, t }
}

function interpolatePose(
  a: Record<string, Keypoint>,
  b: Record<string, Keypoint>,
  t: number
): Record<string, Keypoint> {
  const result: Record<string, Keypoint> = {}
  for (const name of Object.keys(a)) {
    const ka = a[name]
    const kb = b[name]
    if (!kb) {
      result[name] = ka
      continue
    }
    result[name] = {
      x: ka.x + (kb.x - ka.x) * t,
      y: ka.y + (kb.y - ka.y) * t,
      confidence: Math.min(ka.confidence, kb.confidence),
    }
  }
  return result
}

function interpolateBladeState(a: BladeState, b: BladeState, t: number): BladeState {
  return {
    tip_xyz: {
      x: a.tip_xyz.x + (b.tip_xyz.x - a.tip_xyz.x) * t,
      y: a.tip_xyz.y + (b.tip_xyz.y - a.tip_xyz.y) * t,
      z: 0,
    },
    nominal_xyz: a.nominal_xyz,
    velocity_xyz: a.velocity_xyz,
    speed: a.speed,
  }
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
  const rafRef = useRef<number | null>(null)

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
    framesRef.current = frames
  }, [frames])

  useEffect(() => {
    if (!boutId) return
    getBout(Number(boutId)).then((data) => {
      if (data.video_url) setVideoUrl(data.video_url)
      if ((data as any).analysis) setAnalysis((data as any).analysis)
      if (data.frames) setFrames(data.frames)
    })
  }, [boutId])

  useEffect(() => {
    const onFsChange = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', onFsChange)
    return () => document.removeEventListener('fullscreenchange', onFsChange)
  }, [])

  const renderSkeleton = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    const currentFrames = framesRef.current
    if (!video || !canvas || currentFrames.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Sync canvas size with video
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth || video.clientWidth
      canvas.height = video.videoHeight || video.clientHeight
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const currentTimeMs = video.currentTime * 1000
    const interval = findFrameInterval(currentFrames, currentTimeMs)
    if (!interval) return

    const { a, b, t } = interval

    // Draw fencer skeleton (orange)
    if (a.fencer_pose) {
      const pose = b.fencer_pose
        ? interpolatePose(a.fencer_pose, b.fencer_pose, t)
        : a.fencer_pose
      drawSkeleton(ctx, pose, canvas.width, canvas.height, '#f97316')
    }

    // Draw blade overlay (green)
    if (a.blade_state && a.fencer_pose) {
      const bladeState = b.blade_state
        ? interpolateBladeState(a.blade_state, b.blade_state, t)
        : a.blade_state
      const pose = b.fencer_pose
        ? interpolatePose(a.fencer_pose, b.fencer_pose, t)
        : a.fencer_pose
      drawBlade(ctx, pose, bladeState, canvas.width, canvas.height)
    }

    // Draw opponent skeleton (blue)
    if (a.opponent_pose) {
      const pose = b.opponent_pose
        ? interpolatePose(a.opponent_pose, b.opponent_pose, t)
        : a.opponent_pose
      drawSkeleton(ctx, pose, canvas.width, canvas.height, '#3b82f6')
    }
  }, [])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const ro = new ResizeObserver(() => renderSkeleton())
    ro.observe(container)
    return () => ro.disconnect()
  }, [renderSkeleton])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const startLoop = () => {
      const loop = () => {
        renderSkeleton()
        rafRef.current = requestAnimationFrame(loop)
      }
      rafRef.current = requestAnimationFrame(loop)
    }

    const stopLoop = () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
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
            className={`relative bg-black rounded-xl overflow-hidden aspect-video ${isFullscreen ? 'rounded-none' : ''}`}
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
              className="absolute inset-0 w-full h-full pointer-events-none"
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
