import { useParams, useNavigate, Link } from 'react-router-dom'
import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { getBout, getBoutFrames, deleteBout, Frame, Keypoint, BladeState } from '@/api/bouts'
import { drawSkeleton } from '@/utils/skeleton'
import { Trash2, Maximize2, Minimize2, Activity } from 'lucide-react'

interface AnalysisSummary {
  llm_summary: string
  technique_scores: Record<string, number>
}

interface Action {
  id: number
  subject: string
  type: string
  start_ms: number
  end_ms: number
  confidence: number | null
}

const ACTION_COLORS: Record<string, string> = {
  lunge: '#ef4444',
  advance: '#22c55e',
  retreat: '#3b82f6',
  en_garde: '#6b7280',
  fleche: '#f59e0b',
  step_lunge: '#ec4899',
  check_step: '#a78bfa',
  recovery: '#14b8a6',
}

const ACTION_LABELS: Record<string, string> = {
  lunge: 'Lunge',
  advance: 'Advance',
  retreat: 'Retreat',
  en_garde: 'En Garde',
  fleche: 'Fleche',
  step_lunge: 'Step-Lunge',
  check_step: 'Check Step',
  recovery: 'Recovery',
}

function formatMs(ms: number): string {
  const totalSec = Math.floor(ms / 1000)
  const min = Math.floor(totalSec / 60)
  const sec = totalSec % 60
  const millis = Math.floor(ms % 1000)
  return `${min}:${String(sec).padStart(2, '0')}.${String(millis).padStart(3, '0')}`
}

const SPEEDS = [0.25, 0.5, 1, 2]

const TRAIL_LENGTH = 30

interface TipTrailPoint {
  x: number
  y: number
  timestamp_ms: number
  confidence: number
}

function inBounds(x: number, y: number): boolean {
  return x >= -0.05 && x <= 1.05 && y >= -0.05 && y <= 1.05
}

function drawBlade(
  ctx: CanvasRenderingContext2D,
  pose: Record<string, Keypoint>,
  bladeState: BladeState,
  width: number,
  height: number,
  color: string = '#22c55e'
) {
  const wrist = pose['right_wrist'] ?? pose['left_wrist']
  if (!wrist || wrist.confidence < 0.3) return

  const tip = bladeState.tip_xyz
  // Skip if tip is wildly out of frame (bad geometric projection)
  if (!inBounds(tip.x, tip.y)) return

  const conf = bladeState.confidence ?? 0.85
  const alpha = 0.3 + 0.55 * conf

  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.globalAlpha = alpha
  ctx.beginPath()
  ctx.moveTo(wrist.x * width, wrist.y * height)
  ctx.lineTo(tip.x * width, tip.y * height)
  ctx.stroke()

  ctx.fillStyle = color
  ctx.globalAlpha = alpha
  ctx.beginPath()
  ctx.arc(tip.x * width, tip.y * height, 5, 0, Math.PI * 2)
  ctx.fill()
  ctx.globalAlpha = 1
}

function drawTipTrail(
  ctx: CanvasRenderingContext2D,
  trail: TipTrailPoint[],
  width: number,
  height: number
) {
  if (trail.length < 2) return

  for (let i = 1; i < trail.length; i++) {
    const prev = trail[i - 1]
    const curr = trail[i]

    // Skip segments with wild jumps (>20% of frame) or out-of-bounds points
    if (!inBounds(prev.x, prev.y) || !inBounds(curr.x, curr.y)) continue
    const dx = curr.x - prev.x
    const dy = curr.y - prev.y
    if (dx * dx + dy * dy > 0.04) continue  // 0.2^2 = 0.04

    // Progress from 0 (oldest) to 1 (newest)
    const progress = i / (trail.length - 1)
    const trailConf = curr.confidence ?? 1.0

    ctx.strokeStyle = `rgba(34, 197, 94, ${progress * 0.9 * trailConf})`
    ctx.lineWidth = 1 + progress * 2
    ctx.beginPath()
    ctx.moveTo(prev.x * width, prev.y * height)
    ctx.lineTo(curr.x * width, curr.y * height)
    ctx.stroke()
  }
}

function findFrameInterval(
  frames: Frame[],
  timestampMs: number
): { a: Frame; b: Frame; t: number; loIndex: number } | null {
  if (frames.length === 0) return null
  if (frames.length === 1) return { a: frames[0], b: frames[0], t: 0, loIndex: 0 }

  let lo = 0
  let hi = frames.length - 1

  if (timestampMs <= frames[lo].timestamp_ms) return { a: frames[lo], b: frames[lo], t: 0, loIndex: 0 }
  if (timestampMs >= frames[hi].timestamp_ms) return { a: frames[hi], b: frames[hi], t: 0, loIndex: hi }

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
  return { a, b, t, loIndex: lo }
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
      z: ka.z + (kb.z - ka.z) * t,
      confidence: Math.min(ka.confidence, kb.confidence),
    }
  }
  return result
}

function interpolateBladeState(a: BladeState, b: BladeState, t: number): BladeState {
  const confA = a.confidence ?? 1.0
  const confB = b.confidence ?? 1.0
  return {
    tip_xyz: {
      x: a.tip_xyz.x + (b.tip_xyz.x - a.tip_xyz.x) * t,
      y: a.tip_xyz.y + (b.tip_xyz.y - a.tip_xyz.y) * t,
      z: 0,
    },
    nominal_xyz: a.nominal_xyz,
    velocity_xyz: a.velocity_xyz,
    speed: a.speed,
    confidence: confA + (confB - confA) * t,
  }
}

export default function VideoReview() {
  const { boutId } = useParams<{ boutId: string }>()
  const navigate = useNavigate()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const playerPanelRef = useRef<HTMLDivElement>(null)
  const [analysis, setAnalysis] = useState<AnalysisSummary | null>(null)
  const [videoUrl, setVideoUrl] = useState('')
  const [frames, setFrames] = useState<Frame[]>([])
  const [actions, setActions] = useState<Action[]>([])
  const [speed, setSpeed] = useState(1)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [currentTimeMs, setCurrentTimeMs] = useState(0)
  const [videoDurationMs, setVideoDurationMs] = useState(0)
  const [hoveredAction, setHoveredAction] = useState<Action | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 })
  const [showFencerSkeleton, setShowFencerSkeleton] = useState(true)
  const [showOpponentSkeleton, setShowOpponentSkeleton] = useState(true)
  const [showBlade, setShowBlade] = useState(true)
  const [liveBladeState, setLiveBladeState] = useState<BladeState | null>(null)
  const [liveFrameIndex, setLiveFrameIndex] = useState(0)
  const framesRef = useRef<Frame[]>([])
  const rafRef = useRef<number | null>(null)
  const tipTrailRef = useRef<TipTrailPoint[]>([])
  const timelineRef = useRef<HTMLDivElement>(null)
  const scrubberRef = useRef<HTMLDivElement>(null)
  const showFencerRef = useRef(true)
  const showOpponentRef = useRef(true)
  const showBladeRef = useRef(true)
  const [scrubberDragging, setScrubberDragging] = useState(false)
  const [scrubberHover, setScrubberHover] = useState(false)
  const [scrubberHoverFraction, setScrubberHoverFraction] = useState(0)


  function handleSpeed(s: number) {
    setSpeed(s)
    if (videoRef.current) videoRef.current.playbackRate = s
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      playerPanelRef.current?.requestFullscreen()
    } else {
      document.exitFullscreen()
    }
  }

  const scrubberSeek = useCallback((clientX: number) => {
    const bar = scrubberRef.current
    const video = videoRef.current
    if (!bar || !video || !video.duration || !isFinite(video.duration)) return
    const rect = bar.getBoundingClientRect()
    const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    video.currentTime = fraction * video.duration
  }, [])

  const scrubberUpdateHover = useCallback((clientX: number) => {
    const bar = scrubberRef.current
    if (!bar) return
    const rect = bar.getBoundingClientRect()
    const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    setScrubberHoverFraction(fraction)
  }, [])

  // Global mousemove/mouseup for scrubber dragging
  useEffect(() => {
    if (!scrubberDragging) return

    const onMouseMove = (e: MouseEvent) => {
      scrubberSeek(e.clientX)
      scrubberUpdateHover(e.clientX)
    }
    const onMouseUp = () => {
      setScrubberDragging(false)
      setScrubberHover(false)
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [scrubberDragging, scrubberSeek, scrubberUpdateHover])


  useEffect(() => {
    framesRef.current = frames
  }, [frames])

  useEffect(() => { showFencerRef.current = showFencerSkeleton }, [showFencerSkeleton])
  useEffect(() => { showOpponentRef.current = showOpponentSkeleton }, [showOpponentSkeleton])
  useEffect(() => { showBladeRef.current = showBlade }, [showBlade])

  useEffect(() => {
    if (!boutId) return
    getBout(Number(boutId)).then((data) => {
      if (data.video_url) setVideoUrl(data.video_url)
      if (data.analysis) setAnalysis(data.analysis as unknown as AnalysisSummary)
    })
    getBoutFrames(Number(boutId)).then((data) => {
      if (data.frames) setFrames(data.frames)
      if (data.actions) setActions(data.actions)
    }).catch(() => { /* frames not yet available */ })
  }, [boutId])

  useEffect(() => {
    const onFsChange = () => {
      const fsEl = document.fullscreenElement
      setIsFullscreen(!!fsEl)
      // If the video itself went fullscreen (native controls), redirect to
      // the player panel so canvas overlay and controls stay visible
      if (fsEl === videoRef.current && playerPanelRef.current) {
        document.exitFullscreen().then(() => {
          playerPanelRef.current?.requestFullscreen()
        })
      }
    }
    document.addEventListener('fullscreenchange', onFsChange)
    return () => document.removeEventListener('fullscreenchange', onFsChange)
  }, [])

  const renderSkeleton = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    const container = containerRef.current
    const currentFrames = framesRef.current
    if (!video || !canvas || !container || currentFrames.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Sync canvas internal resolution with video native resolution
    const vw = video.videoWidth || video.clientWidth
    const vh = video.videoHeight || video.clientHeight
    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw
      canvas.height = vh
    }

    // Compute where the video content renders within the container
    // (object-contain creates letterboxing when aspect ratios differ)
    const containerW = container.clientWidth
    const containerH = container.clientHeight
    const videoAR = vw / vh
    const containerAR = containerW / containerH
    let renderW: number, renderH: number, offsetX: number, offsetY: number
    if (videoAR > containerAR) {
      renderW = containerW
      renderH = containerW / videoAR
      offsetX = 0
      offsetY = (containerH - renderH) / 2
    } else {
      renderH = containerH
      renderW = containerH * videoAR
      offsetX = (containerW - renderW) / 2
      offsetY = 0
    }
    canvas.style.left = `${offsetX}px`
    canvas.style.top = `${offsetY}px`
    canvas.style.width = `${renderW}px`
    canvas.style.height = `${renderH}px`

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Update timeline playhead position
    const timeMs = video.currentTime * 1000
    setCurrentTimeMs(timeMs)
    if (video.duration && isFinite(video.duration)) {
      setVideoDurationMs(video.duration * 1000)
    }

    const interval = findFrameInterval(currentFrames, timeMs)
    if (!interval) return

    const { a, b, t, loIndex } = interval

    // Update live frame data for the stats panel
    setLiveFrameIndex(loIndex)
    if (a.blade_state) {
      const bs = b.blade_state
        ? interpolateBladeState(a.blade_state, b.blade_state, t)
        : a.blade_state
      setLiveBladeState(bs)
    } else {
      setLiveBladeState(null)
    }

    // Draw fencer skeleton (orange)
    if (showFencerRef.current && a.fencer_pose) {
      const pose = b.fencer_pose
        ? interpolatePose(a.fencer_pose, b.fencer_pose, t)
        : a.fencer_pose
      drawSkeleton(ctx, pose, canvas.width, canvas.height, '#f97316')
    }

    // Draw blade overlay (green) + tip trajectory trail
    if (showBladeRef.current && a.blade_state && a.fencer_pose) {
      const bladeState = b.blade_state
        ? interpolateBladeState(a.blade_state, b.blade_state, t)
        : a.blade_state
      const pose = b.fencer_pose
        ? interpolatePose(a.fencer_pose, b.fencer_pose, t)
        : a.fencer_pose
      drawBlade(ctx, pose, bladeState, canvas.width, canvas.height)

      // Update tip trajectory trail buffer
      const tip = bladeState.tip_xyz
      if (inBounds(tip.x, tip.y) && (tip.x !== 0 || tip.y !== 0)) {
        const trail = tipTrailRef.current
        trail.push({ x: tip.x, y: tip.y, timestamp_ms: timeMs, confidence: bladeState.confidence ?? 1.0 })
        if (trail.length > TRAIL_LENGTH) {
          trail.splice(0, trail.length - TRAIL_LENGTH)
        }
      }

      // Draw the trajectory trail
      drawTipTrail(ctx, tipTrailRef.current, canvas.width, canvas.height)
    }

    // Draw opponent skeleton (blue)
    if (showOpponentRef.current && a.opponent_pose) {
      const pose = b.opponent_pose
        ? interpolatePose(a.opponent_pose, b.opponent_pose, t)
        : a.opponent_pose
      drawSkeleton(ctx, pose, canvas.width, canvas.height, '#3b82f6')

      // Draw opponent blade overlay (cyan)
      if (showBladeRef.current && a.opponent_blade_state) {
        const oppBlade = b.opponent_blade_state
          ? interpolateBladeState(a.opponent_blade_state, b.opponent_blade_state, t)
          : a.opponent_blade_state
        drawBlade(ctx, pose, oppBlade, canvas.width, canvas.height, '#06b6d4')
      }
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
    const handleSeeked = () => {
      tipTrailRef.current = []
      renderSkeleton()
    }
    const handleLoadedMetadata = () => {
      renderSkeleton()
      if (video.duration && isFinite(video.duration)) {
        setVideoDurationMs(video.duration * 1000)
      }
    }

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

  // Derive the currently active action from video timestamp
  const activeAction = useMemo(() => {
    return actions.find(a => a.subject !== 'opponent' && currentTimeMs >= a.start_ms && currentTimeMs <= a.end_ms) ?? null
  }, [actions, currentTimeMs])

  // Speed color coding: green (<1), yellow (1-3), red (>3)
  function speedColor(spd: number | null): string {
    if (spd == null) return 'text-gray-500'
    if (spd < 1) return 'text-green-400'
    if (spd <= 3) return 'text-yellow-400'
    return 'text-red-400'
  }

  return (
    <div className="max-w-screen-2xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold">Bout Review — #{boutId}</h1>
        <div className="flex items-center gap-3">
          <Link
            to={`/bouts/${boutId}/drill`}
            className="flex items-center gap-2 px-4 py-2 bg-brand-500 hover:bg-sky-400 rounded-lg text-sm font-medium transition-colors"
          >
            <Activity size={14} /> Drill Report
          </Link>
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
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Video + skeleton canvas + controls */}
        <div
          ref={playerPanelRef}
          className={[
            'lg:col-span-3',
            isFullscreen
              ? 'flex flex-col bg-gray-950 p-3 gap-3'
              : 'space-y-3',
          ].join(' ')}
        >
          <div className={[
            'relative',
            isFullscreen ? 'flex-1 min-h-0' : '',
          ].join(' ')}>
            <div
              ref={containerRef}
              className={[
                'relative bg-black overflow-hidden',
                isFullscreen ? 'h-full rounded-lg' : 'rounded-xl',
              ].join(' ')}
              style={isFullscreen ? {} : {
                resize: 'both',
                aspectRatio: '16 / 9',
                maxWidth: '100%',
                minWidth: '320px',
                minHeight: '180px',
              }}
            >
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                muted
                className="w-full h-full object-contain"
              />
              <canvas
                ref={canvasRef}
                className="absolute pointer-events-none"
              />
            </div>
            <button
              onClick={toggleFullscreen}
              className="absolute top-2 right-2 z-20 p-1.5 bg-black/50 hover:bg-black/70 rounded-md text-white transition-colors"
              title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen with overlay'}
            >
              {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            </button>
          </div>
          {/* Custom Scrubber Bar */}
          {videoDurationMs > 0 && (
            <div
              className="relative group px-1"
              style={{ paddingTop: '10px', paddingBottom: '2px' }}
            >
              {/* Time tooltip */}
              {(scrubberDragging || scrubberHover) && (
                <div
                  className="absolute bottom-full mb-1.5 px-1.5 py-0.5 bg-gray-950 border border-gray-700 rounded text-[11px] text-gray-200 font-mono pointer-events-none whitespace-nowrap z-20"
                  style={{
                    left: `${(scrubberDragging
                      ? (currentTimeMs / videoDurationMs)
                      : scrubberHoverFraction
                    ) * 100}%`,
                    transform: 'translateX(-50%)',
                  }}
                >
                  {formatMs(
                    scrubberDragging
                      ? currentTimeMs
                      : scrubberHoverFraction * videoDurationMs
                  )}
                </div>
              )}
              {/* Track */}
              <div
                ref={scrubberRef}
                className="relative w-full h-1.5 bg-gray-700 rounded-full cursor-pointer"
                onMouseDown={(e) => {
                  e.preventDefault()
                  setScrubberDragging(true)
                  scrubberSeek(e.clientX)
                  scrubberUpdateHover(e.clientX)
                }}
                onMouseEnter={() => setScrubberHover(true)}
                onMouseLeave={() => { if (!scrubberDragging) setScrubberHover(false) }}
                onMouseMove={(e) => { if (!scrubberDragging) scrubberUpdateHover(e.clientX) }}
              >
                {/* Filled portion */}
                <div
                  className="absolute inset-y-0 left-0 bg-white rounded-full pointer-events-none"
                  style={{
                    width: `${(currentTimeMs / videoDurationMs) * 100}%`,
                  }}
                />
                {/* Thumb */}
                <div
                  className="absolute top-1/2 -translate-y-1/2 pointer-events-none z-10"
                  style={{
                    left: `${(currentTimeMs / videoDurationMs) * 100}%`,
                  }}
                >
                  <div
                    className={[
                      'w-3 h-3 -ml-1.5 bg-white rounded-full shadow-md transition-transform duration-100',
                      scrubberDragging || scrubberHover ? 'scale-125' : 'scale-0 group-hover:scale-100',
                    ].join(' ')}
                  />
                </div>
              </div>
            </div>
          )}
          {/* Action Timeline */}
          {actions.length > 0 && videoDurationMs > 0 && (() => {
            const fencerActions = actions.filter(a => a.subject !== 'opponent')
            const opponentActions = actions.filter(a => a.subject === 'opponent')
            const hasOpponent = opponentActions.length > 0

            const renderTrack = (trackActions: Action[], label: string, trackRef?: React.RefObject<HTMLDivElement>) => (
              <div
                ref={trackRef}
                className="relative h-6 bg-gray-800 rounded-md cursor-pointer overflow-hidden"
                onClick={(e) => {
                  const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect()
                  const fraction = (e.clientX - rect.left) / rect.width
                  const seekMs = fraction * videoDurationMs
                  if (videoRef.current) {
                    videoRef.current.currentTime = seekMs / 1000
                  }
                }}
              >
                {/* Track label */}
                <span className="absolute left-1.5 top-1/2 -translate-y-1/2 text-[9px] font-medium text-gray-500 z-10 pointer-events-none">
                  {label}
                </span>
                {/* Action segments */}
                {trackActions.map((action) => {
                  const left = (action.start_ms / videoDurationMs) * 100
                  const width = ((action.end_ms - action.start_ms) / videoDurationMs) * 100
                  const color = ACTION_COLORS[action.type] ?? '#9ca3af'
                  return (
                    <div
                      key={action.id}
                      className="absolute top-1 bottom-1 rounded-sm transition-opacity hover:opacity-100"
                      style={{
                        left: `${left}%`,
                        width: `${Math.max(width, 0.3)}%`,
                        backgroundColor: color,
                        opacity: 0.85,
                      }}
                      onClick={(e) => {
                        e.stopPropagation()
                        if (videoRef.current) {
                          videoRef.current.currentTime = action.start_ms / 1000
                        }
                      }}
                      onMouseEnter={(e) => {
                        setHoveredAction(action)
                        const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect()
                        setTooltipPos({ x: rect.left + rect.width / 2, y: rect.top })
                      }}
                      onMouseLeave={() => setHoveredAction(null)}
                    />
                  )
                })}
                {/* Current time playhead */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-white z-10 pointer-events-none"
                  style={{
                    left: `${(currentTimeMs / videoDurationMs) * 100}%`,
                  }}
                />
              </div>
            )

            return (
              <div className="bg-gray-900 rounded-xl p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                    Action Timeline
                  </h3>
                  <div className="flex items-center gap-3">
                    {Object.entries(ACTION_COLORS).map(([type, color]) => (
                      <div key={type} className="flex items-center gap-1">
                        <span
                          className="inline-block w-2.5 h-2.5 rounded-sm"
                          style={{ backgroundColor: color }}
                        />
                        <span className="text-[10px] text-gray-500">
                          {ACTION_LABELS[type] ?? type}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="space-y-1">
                  {renderTrack(fencerActions, 'You', timelineRef)}
                  {hasOpponent && renderTrack(opponentActions, 'Opp')}
                </div>
                {/* Tooltip */}
                {hoveredAction && (
                  <div
                    className="fixed z-50 px-2.5 py-1.5 bg-gray-950 border border-gray-700 rounded-md shadow-lg pointer-events-none"
                    style={{
                      left: `${tooltipPos.x}px`,
                      top: `${tooltipPos.y - 8}px`,
                      transform: 'translate(-50%, -100%)',
                    }}
                  >
                    <div className="flex items-center gap-1.5">
                      <span
                        className="inline-block w-2 h-2 rounded-sm"
                        style={{ backgroundColor: ACTION_COLORS[hoveredAction.type] ?? '#9ca3af' }}
                      />
                      <span className="text-xs font-medium text-gray-200">
                        {ACTION_LABELS[hoveredAction.type] ?? hoveredAction.type}
                      </span>
                      <span className="text-[10px] text-gray-500">
                        ({hoveredAction.subject === 'opponent' ? 'Opponent' : 'You'})
                      </span>
                    </div>
                    <p className="text-[10px] text-gray-400 mt-0.5">
                      {formatMs(hoveredAction.start_ms)} — {formatMs(hoveredAction.end_ms)}
                      {hoveredAction.confidence != null && (
                        <span className="ml-1.5 text-gray-500">
                          ({Math.round(hoveredAction.confidence * 100)}%)
                        </span>
                      )}
                    </p>
                  </div>
                )}
              </div>
            )
          })()}
          <div className="flex items-center gap-4 flex-wrap">
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
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-gray-500">Overlays</span>
              <button
                onClick={() => setShowFencerSkeleton(v => !v)}
                className={[
                  'px-2 py-0.5 rounded-full text-[11px] font-medium transition-colors',
                  showFencerSkeleton
                    ? 'bg-orange-500 text-white'
                    : 'bg-gray-800 text-gray-500 hover:bg-gray-700',
                ].join(' ')}
              >
                You
              </button>
              <button
                onClick={() => setShowOpponentSkeleton(v => !v)}
                className={[
                  'px-2 py-0.5 rounded-full text-[11px] font-medium transition-colors',
                  showOpponentSkeleton
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-800 text-gray-500 hover:bg-gray-700',
                ].join(' ')}
              >
                Opponent
              </button>
              <button
                onClick={() => setShowBlade(v => !v)}
                className={[
                  'px-2 py-0.5 rounded-full text-[11px] font-medium transition-colors',
                  showBlade
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-800 text-gray-500 hover:bg-gray-700',
                ].join(' ')}
              >
                Blade
              </button>
            </div>
          </div>
          <p className="text-xs text-gray-500">
            {frames.length > 0
              ? `Skeleton overlay active — ${frames.length} frames loaded`
              : 'No pose data available'}
          </p>
        </div>

        {/* Analysis panel */}
        <div className="lg:col-span-1 space-y-4">
          {/* Live Frame Data */}
          <div className="bg-gray-800 rounded-xl p-4">
            <h2 className="font-semibold mb-3 text-sm text-gray-400 uppercase tracking-wider">
              Frame Data
            </h2>
            {frames.length > 0 ? (
              <div className="space-y-3">
                {/* Frame Number */}
                <div className="flex justify-between items-baseline">
                  <span className="text-xs text-gray-400">Frame</span>
                  <span className="text-sm text-white font-mono">
                    {liveFrameIndex + 1} / {frames.length}
                  </span>
                </div>

                {/* Tip Speed */}
                <div className="flex justify-between items-baseline">
                  <span className="text-xs text-gray-400">Tip Speed</span>
                  {liveBladeState?.speed != null ? (
                    <span className={`text-sm font-mono font-medium ${speedColor(liveBladeState.speed)}`}>
                      {liveBladeState.speed.toFixed(1)} m/s
                    </span>
                  ) : (
                    <span className="text-sm text-gray-500 font-mono">{'\u2014'}</span>
                  )}
                </div>

                {/* Blade Confidence */}
                <div className="flex justify-between items-baseline">
                  <span className="text-xs text-gray-400">Blade Conf</span>
                  {liveBladeState?.confidence != null ? (
                    <span className="text-sm font-mono font-medium text-gray-200">
                      {Math.round(liveBladeState.confidence * 100)}%
                    </span>
                  ) : (
                    <span className="text-sm text-gray-500 font-mono">{'\u2014'}</span>
                  )}
                </div>

                {/* Tip Position */}
                <div className="flex justify-between items-baseline">
                  <span className="text-xs text-gray-400">Tip Position</span>
                  {liveBladeState ? (
                    <span className="text-sm text-white font-mono">
                      ({liveBladeState.tip_xyz.x.toFixed(3)}, {liveBladeState.tip_xyz.y.toFixed(3)})
                    </span>
                  ) : (
                    <span className="text-sm text-gray-500 font-mono">{'\u2014'}</span>
                  )}
                </div>

                {/* Current Action */}
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-400">Action</span>
                  {activeAction ? (
                    <span
                      className="text-xs font-medium px-2 py-0.5 rounded-full"
                      style={{
                        backgroundColor: (ACTION_COLORS[activeAction.type] ?? '#6b7280') + '33',
                        color: ACTION_COLORS[activeAction.type] ?? '#9ca3af',
                        border: `1px solid ${ACTION_COLORS[activeAction.type] ?? '#6b7280'}66`,
                      }}
                    >
                      {ACTION_LABELS[activeAction.type] ?? activeAction.type}
                    </span>
                  ) : (
                    <span className="text-sm text-gray-500 font-mono">{'\u2014'}</span>
                  )}
                </div>

                {/* Timestamp */}
                <div className="flex justify-between items-baseline">
                  <span className="text-xs text-gray-400">Time</span>
                  <span className="text-sm text-white font-mono">
                    {formatMs(currentTimeMs)}
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-500">No frame data available.</p>
            )}
          </div>

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
