import { useEffect, useRef, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getBoutStatus, deleteBout, type BoutStatus } from '@/api/bouts'
import { CheckCircle, Loader2, Circle, XCircle } from 'lucide-react'

// ---------------------------------------------------------------------------
// Stage definitions
// ---------------------------------------------------------------------------

interface StageDef {
  key: string
  label: string
  description: string
  weight: number
}

const STAGE_DEFS: StageDef[] = [
  { key: 'ingest',                label: 'Ingesting video',        description: 'FFprobe metadata extraction',                weight: 5  },
  { key: 'pose_estimation',       label: 'Pose Estimation',        description: 'YOLOv8 skeleton detection · GPU-intensive',  weight: 60 },
  { key: 'blade_tracking',        label: 'Blade Tracking',         description: 'Epee tip detection and trajectory tracking', weight: 8  },
  { key: 'action_classification', label: 'Action Classification',  description: 'Classifying fencing actions and footwork',   weight: 7  },
  { key: 'llm_synthesis',         label: 'AI Coaching',            description: 'Generating feedback via Claude API',         weight: 20 },
]

const TOTAL_WEIGHT = STAGE_DEFS.reduce((s, d) => s + d.weight, 0)

const STAGE_ORDER = STAGE_DEFS.map(d => d.key)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Return within-stage progress (0–100) for the active stage. */
function stageProgress(
  stageKey: string,
  pct: number,
  frame?: number,
  total_frames?: number,
): number {
  switch (stageKey) {
    case 'ingest':
      return 100
    case 'pose_estimation':
      if (frame != null && total_frames && total_frames > 0) {
        return Math.min(100, Math.round((frame / total_frames) * 100))
      }
      return Math.min(100, Math.max(0, Math.round(((pct - 20) / 60) * 100)))
    case 'blade_tracking':
      return Math.min(100, Math.max(0, Math.round(((pct - 80) / 8) * 100)))
    case 'action_classification':
      return Math.min(100, Math.max(0, Math.round(((pct - 88) / 7) * 100)))
    case 'llm_synthesis':
      return Math.min(100, Math.max(0, Math.round(((pct - 95) / 20) * 100)))
    default:
      return 0
  }
}

/** Determine whether a stage is complete, active, or pending given current stage. */
function stageStatus(
  stageKey: string,
  currentStage: string,
): 'complete' | 'active' | 'pending' {
  if (currentStage === 'complete') return 'complete'
  const currentIdx = STAGE_ORDER.indexOf(currentStage)
  const thisIdx = STAGE_ORDER.indexOf(stageKey)
  if (thisIdx < currentIdx) return 'complete'
  if (thisIdx === currentIdx) return 'active'
  return 'pending'
}

function formatEta(seconds: number): string {
  if (seconds < 60) return `~${Math.round(seconds)}s remaining`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return s > 0 ? `~${m}m ${s}s remaining` : `~${m}m remaining`
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ResourceBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center gap-2 text-xs text-gray-400">
      <span className="w-8 shrink-0">{label}</span>
      <div className="relative flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 bg-cyan-500 rounded-full transition-all duration-500"
          style={{ width: `${value}%` }}
        />
      </div>
      <span className="w-7 text-right tabular-nums">{value}%</span>
    </div>
  )
}

interface StageRowProps {
  def: StageDef
  status: 'complete' | 'active' | 'pending'
  trackWidthPct: number
  withinPct: number
  frame?: number
  total_frames?: number
  gpu_mem_pct?: number
  cpu_pct?: number
  eta?: number | null
}

function StageRow({
  def,
  status,
  trackWidthPct,
  withinPct,
  frame,
  total_frames,
  gpu_mem_pct,
  cpu_pct,
  eta,
}: StageRowProps) {
  const isComplete = status === 'complete'
  const isActive   = status === 'active'
  const isPending  = status === 'pending'

  return (
    <div
      className={[
        'flex flex-col gap-1.5 rounded-xl px-4 py-3 transition-colors duration-300',
        isActive  ? 'bg-gray-800/70 ring-1 ring-brand-500/40' : '',
        isComplete ? 'bg-gray-800/30' : '',
        isPending  ? 'opacity-50' : '',
      ].join(' ')}
    >
      {/* Top row: icon + label + track */}
      <div className="flex items-center gap-3">
        {/* Status icon */}
        <div className="shrink-0 w-5 flex justify-center">
          {isComplete && <CheckCircle size={18} className="text-green-400" />}
          {isActive   && <Loader2    size={18} className="text-brand-400 animate-spin" />}
          {isPending  && <Circle     size={18} className="text-gray-600" />}
        </div>

        {/* Label + description */}
        <div className="min-w-0 flex-1">
          <p className={[
            'text-sm font-medium leading-tight',
            isComplete ? 'text-green-300' : isActive ? 'text-white' : 'text-gray-400',
          ].join(' ')}>
            {def.label}
          </p>
          <p className="text-xs text-gray-500 truncate">{def.description}</p>
        </div>

        {/* Relative-weight progress track */}
        <div
          className="relative h-3 rounded-full overflow-hidden bg-gray-700 shrink-0"
          style={{ width: `${trackWidthPct}%` }}
          title={`Relative cost: ${def.weight} / ${TOTAL_WEIGHT}`}
        >
          {isComplete && (
            <div className="absolute inset-0 bg-green-500/40 rounded-full" />
          )}
          {isActive && (
            <div
              className="absolute inset-y-0 left-0 bg-brand-500 rounded-full transition-all duration-500"
              style={{ width: `${withinPct}%` }}
            />
          )}
          {/* pct label inside track when active */}
          {isActive && (
            <span className="absolute inset-0 flex items-center justify-center text-[10px] font-semibold text-white/80 pointer-events-none select-none">
              {withinPct}%
            </span>
          )}
        </div>
      </div>

      {/* Frame counter (pose_estimation only, when active) */}
      {isActive && frame != null && total_frames != null && (
        <p className="text-xs text-gray-400 pl-8">
          Frame {frame.toLocaleString()} / {total_frames.toLocaleString()}
        </p>
      )}

      {/* ETA (pose_estimation only, when active and eta is available) */}
      {isActive && eta != null && def.key === 'pose_estimation' && (
        <p className="text-xs text-brand-400 pl-8">{formatEta(eta)}</p>
      )}

      {/* Resource bars (active only, when data present) */}
      {isActive && (gpu_mem_pct != null || cpu_pct != null) && (
        <div className="pl-8 flex flex-col gap-1">
          {gpu_mem_pct != null && <ResourceBar label="GPU"  value={gpu_mem_pct} />}
          {cpu_pct     != null && <ResourceBar label="CPU"  value={cpu_pct}     />}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Pipeline visualization
// ---------------------------------------------------------------------------

interface PipelineProps {
  pipeline: BoutStatus['pipeline_progress']
  eta?: number | null
}

function PipelineVisualization({ pipeline, eta }: PipelineProps) {
  const currentStage   = pipeline.stage ?? ''
  const pct            = pipeline.pct ?? 0
  const frame          = pipeline.frame
  const total_frames   = pipeline.total_frames
  const gpu_mem_pct    = pipeline.gpu_mem_pct
  const cpu_pct        = pipeline.cpu_pct

  return (
    <div className="flex flex-col gap-1">
      {STAGE_DEFS.map(def => {
        const ss            = stageStatus(def.key, currentStage)
        const trackWidthPct = Math.round((def.weight / TOTAL_WEIGHT) * 100)
        const withinPct     = ss === 'active'
          ? stageProgress(def.key, pct, frame, total_frames)
          : 0

        return (
          <StageRow
            key={def.key}
            def={def}
            status={ss}
            trackWidthPct={trackWidthPct}
            withinPct={withinPct}
            frame={ss === 'active' ? frame : undefined}
            total_frames={ss === 'active' ? total_frames : undefined}
            gpu_mem_pct={ss === 'active' ? gpu_mem_pct : undefined}
            cpu_pct={ss === 'active' ? cpu_pct : undefined}
            eta={ss === 'active' ? eta : undefined}
          />
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function ProcessingStatus() {
  const { boutId } = useParams<{ boutId: string }>()
  const navigate = useNavigate()
  const [status, setStatus] = useState<BoutStatus | null>(null)
  const [eta, setEta] = useState<number | null>(null)
  const prevFrameRef = useRef<{ frame: number; time: number } | null>(null)
  const etaRef = useRef<number | null>(null)

  useEffect(() => {
    if (!boutId) return
    const id = Number(boutId)

    const poll = async () => {
      try {
        const s = await getBoutStatus(id)
        setStatus(s)

        if (s.pipeline_progress.frame != null && s.pipeline_progress.frame > 0) {
          const now = Date.now()
          const frame = s.pipeline_progress.frame!
          const total = s.pipeline_progress.total_frames!

          if (prevFrameRef.current && frame > prevFrameRef.current.frame) {
            const elapsed = (now - prevFrameRef.current.time) / 1000
            const framesDone = frame - prevFrameRef.current.frame
            const fps = framesDone / elapsed
            const remaining = (total - frame) / fps
            etaRef.current = remaining
          }
          prevFrameRef.current = { frame, time: now }
          setEta(etaRef.current)
        }

        if (s.status === 'complete') {
          setTimeout(() => navigate(`/bouts/${boutId}/review`), 1500)
        }
      } catch {
        // retry silently
      }
    }

    poll()
    const interval = setInterval(poll, 2000)
    return () => clearInterval(interval)
  }, [boutId, navigate])

  return (
    <div className="max-w-lg mx-auto mt-20 space-y-6">
      {status?.status === 'failed' ? (
        <div className="text-center space-y-3">
          <XCircle className="mx-auto text-red-400" size={48} />
          <p className="text-red-400 font-medium">Analysis failed</p>
          <p className="text-gray-500 text-sm">{status.error}</p>
        </div>
      ) : status?.status === 'complete' ? (
        <div className="text-center space-y-3">
          <CheckCircle className="mx-auto text-green-400" size={48} />
          <p className="font-medium">Analysis complete — loading review…</p>
        </div>
      ) : (
        <>
          <div className="bg-gray-900 border border-gray-700/50 rounded-2xl p-4 space-y-1">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-1 mb-3">
              Pipeline
            </p>
            <PipelineVisualization pipeline={status?.pipeline_progress ?? {}} eta={eta} />
          </div>

          <div className="text-center">
            <button
              onClick={async () => {
                try {
                  await deleteBout(Number(boutId))
                } catch {
                  // ignore errors
                }
                navigate('/')
              }}
              className="px-6 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm font-medium transition-colors text-gray-300"
            >
              Cancel
            </button>
          </div>
        </>
      )}
    </div>
  )
}
