import { useParams, useNavigate } from 'react-router-dom'
import { useEffect, useRef, useState, useCallback } from 'react'
import {
  configureROI,
  getPreview,
  triggerPreview,
  PreviewFrame,
  PreviewDetection,
  DetectionSelection,
} from '@/api/bouts'
import { drawSkeleton } from '@/utils/skeleton'
import clsx from 'clsx'

interface Selection {
  frameIdx: number
  detIdx: number
}

export default function SelectSkeletons() {
  const { boutId } = useParams<{ boutId: string }>()
  const navigate = useNavigate()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)

  const [previewData, setPreviewData] = useState<{ frames: PreviewFrame[] } | null>(null)
  const [activeFrameIdx, setActiveFrameIdx] = useState(0)
  const [fencerSelection, setFencerSelection] = useState<Selection | null>(null)
  const [opponentSelection, setOpponentSelection] = useState<Selection | null>(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [imgLoaded, setImgLoaded] = useState(false)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Poll for preview data
  useEffect(() => {
    if (!boutId) return

    let cancelled = false

    async function poll() {
      try {
        const resp = await getPreview(Number(boutId))
        if (cancelled) return

        if (resp.status === 'ready' && resp.preview_data) {
          setPreviewData(resp.preview_data)
          setLoading(false)
          setError(null)

          // Auto-select if only 1 detection total
          const allDetections = resp.preview_data.frames.flatMap((f, fi) =>
            f.detections.map((d) => ({ frameIdx: fi, detIdx: d.index }))
          )
          if (allDetections.length === 1) {
            setFencerSelection(allDetections[0])
          }

          if (pollRef.current) {
            clearInterval(pollRef.current)
            pollRef.current = null
          }
        } else if (resp.status === 'failed') {
          setError(resp.error || 'Preview generation failed')
          setLoading(false)
          if (pollRef.current) {
            clearInterval(pollRef.current)
            pollRef.current = null
          }
        }
        // status === 'processing' — keep polling
      } catch (err: any) {
        if (cancelled) return
        console.error('Preview poll error:', err)
      }
    }

    poll()
    pollRef.current = setInterval(poll, 1000)

    return () => {
      cancelled = true
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [boutId])

  // Current frame data
  const activeFrame = previewData?.frames[activeFrameIdx] ?? null
  const imageUrl = activeFrame ? `/uploads/${activeFrame.image_key}` : ''

  // Reset image loaded state when frame changes
  useEffect(() => {
    setImgLoaded(false)
  }, [activeFrameIdx])

  // Draw canvas when image loads or selections change
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img || !activeFrame) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    for (const det of activeFrame.detections) {
      const isFencer =
        fencerSelection &&
        fencerSelection.frameIdx === activeFrameIdx &&
        fencerSelection.detIdx === det.index
      const isOpponent =
        opponentSelection &&
        opponentSelection.frameIdx === activeFrameIdx &&
        opponentSelection.detIdx === det.index

      let color: string
      let opacity: number
      let label: string

      if (isFencer) {
        color = '#f97316'
        opacity = 1.0
        label = 'You'
      } else if (isOpponent) {
        color = '#3b82f6'
        opacity = 1.0
        label = 'Opponent'
      } else {
        color = '#d1d5db'
        opacity = 0.5
        label = `Person ${det.index + 1}`
      }

      // Draw skeleton
      ctx.save()
      ctx.globalAlpha = opacity
      drawSkeleton(ctx, det.keypoints, canvas.width, canvas.height, color)
      ctx.restore()

      // Draw bounding box (dashed)
      const bbox = det.bbox
      const bx = bbox.x1 * canvas.width
      const by = bbox.y1 * canvas.height
      const bw = (bbox.x2 - bbox.x1) * canvas.width
      const bh = (bbox.y2 - bbox.y1) * canvas.height

      ctx.save()
      ctx.globalAlpha = opacity
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.setLineDash([6, 4])
      ctx.strokeRect(bx, by, bw, bh)
      ctx.setLineDash([])
      ctx.restore()

      // Draw label
      ctx.save()
      ctx.globalAlpha = opacity
      const fontSize = Math.max(14, Math.min(20, canvas.width * 0.018))
      ctx.font = `bold ${fontSize}px sans-serif`
      const textMetrics = ctx.measureText(label)
      const textHeight = fontSize
      const padX = 6
      const padY = 4
      const labelX = bx
      const labelY = by - textHeight - padY * 2

      // Background pill
      ctx.fillStyle = color
      ctx.globalAlpha = opacity * 0.85
      ctx.beginPath()
      const pillW = textMetrics.width + padX * 2
      const pillH = textHeight + padY * 2
      const pillY = Math.max(0, labelY)
      ctx.roundRect(labelX, pillY, pillW, pillH, 4)
      ctx.fill()

      // Text
      ctx.globalAlpha = opacity
      ctx.fillStyle = '#ffffff'
      ctx.fillText(label, labelX + padX, pillY + padY + textHeight * 0.85)
      ctx.restore()
    }
  }, [activeFrame, activeFrameIdx, fencerSelection, opponentSelection])

  useEffect(() => {
    if (imgLoaded) draw()
  }, [imgLoaded, draw])

  // Handle canvas click — find which detection was clicked
  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current
    if (!canvas || !activeFrame) return

    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const clickX = ((e.clientX - rect.left) * scaleX) / canvas.width // normalized 0-1
    const clickY = ((e.clientY - rect.top) * scaleY) / canvas.height // normalized 0-1

    // Check which detection bbox contains the click
    let clickedDet: PreviewDetection | null = null
    for (const det of activeFrame.detections) {
      const b = det.bbox
      if (clickX >= b.x1 && clickX <= b.x2 && clickY >= b.y1 && clickY <= b.y2) {
        clickedDet = det
        break
      }
    }

    // If no direct hit, find nearest bbox center within 15% distance
    if (!clickedDet) {
      let minDist = 0.15
      for (const det of activeFrame.detections) {
        const b = det.bbox
        const cx = (b.x1 + b.x2) / 2
        const cy = (b.y1 + b.y2) / 2
        const dist = Math.sqrt((clickX - cx) ** 2 + (clickY - cy) ** 2)
        if (dist < minDist) {
          minDist = dist
          clickedDet = det
        }
      }
    }

    if (!clickedDet) return

    const clicked: Selection = { frameIdx: activeFrameIdx, detIdx: clickedDet.index }

    // Assignment logic
    const isFencer =
      fencerSelection &&
      fencerSelection.frameIdx === clicked.frameIdx &&
      fencerSelection.detIdx === clicked.detIdx
    const isOpponent =
      opponentSelection &&
      opponentSelection.frameIdx === clicked.frameIdx &&
      opponentSelection.detIdx === clicked.detIdx

    if (isFencer) {
      // Clicking assigned fencer deselects it
      setFencerSelection(null)
    } else if (isOpponent) {
      // Clicking assigned opponent deselects it
      setOpponentSelection(null)
    } else if (!fencerSelection) {
      // No fencer yet — assign as fencer
      setFencerSelection(clicked)
    } else if (!opponentSelection) {
      // Fencer set, no opponent — assign as opponent
      setOpponentSelection(clicked)
    } else {
      // Both assigned — replace opponent
      setOpponentSelection(clicked)
    }
  }

  async function handleSubmit() {
    if (!boutId) return
    setSubmitting(true)
    setError(null)
    try {
      const fencerDet: DetectionSelection | undefined = fencerSelection
        ? { frame_index: fencerSelection.frameIdx, detection_index: fencerSelection.detIdx }
        : undefined
      const opponentDet: DetectionSelection | undefined = opponentSelection
        ? { frame_index: opponentSelection.frameIdx, detection_index: opponentSelection.detIdx }
        : undefined

      await configureROI(
        Number(boutId),
        null,
        null,
        fencerDet ?? null,
        opponentDet ?? null,
      )
      navigate(`/bouts/${boutId}/processing`)
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Unknown error'
      setError(msg)
      console.error(err)
      setSubmitting(false)
    }
  }

  async function handleSkip() {
    if (!boutId) return
    setSubmitting(true)
    setError(null)
    try {
      await configureROI(Number(boutId), null, null)
      navigate(`/bouts/${boutId}/processing`)
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Unknown error'
      setError(msg)
      console.error(err)
      setSubmitting(false)
    }
  }

  async function handleRetry() {
    if (!boutId) return
    setLoading(true)
    setError(null)
    try {
      await triggerPreview(Number(boutId))
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to trigger preview')
      setLoading(false)
      return
    }

    // Resume polling
    const poll = async () => {
      try {
        const resp = await getPreview(Number(boutId))
        if (resp.status === 'ready' && resp.preview_data) {
          setPreviewData(resp.preview_data)
          setLoading(false)
          setError(null)

          const allDetections = resp.preview_data.frames.flatMap((f, fi) =>
            f.detections.map((d) => ({ frameIdx: fi, detIdx: d.index }))
          )
          if (allDetections.length === 1) {
            setFencerSelection(allDetections[0])
          }

          if (pollRef.current) {
            clearInterval(pollRef.current)
            pollRef.current = null
          }
        } else if (resp.status === 'failed') {
          setError(resp.error || 'Preview generation failed')
          setLoading(false)
          if (pollRef.current) {
            clearInterval(pollRef.current)
            pollRef.current = null
          }
        }
      } catch (err: any) {
        console.error('Preview poll error:', err)
      }
    }

    pollRef.current = setInterval(poll, 1000)
  }

  // Count total detections
  const totalDetections =
    previewData?.frames.reduce((sum, f) => sum + f.detections.length, 0) ?? 0

  // Loading state
  if (loading) {
    return (
      <div className="max-w-screen-xl mx-auto space-y-6">
        <div>
          <h1 className="text-xl font-bold">Select Fencers</h1>
          <p className="text-sm text-gray-400 mt-1">Preparing preview frames...</p>
        </div>
        <div className="flex flex-col items-center justify-center h-64 gap-4">
          <div className="animate-spin rounded-full h-10 w-10 border-2 border-gray-600 border-t-brand-500" />
          <p className="text-gray-400 text-sm">Detecting fencers...</p>
        </div>
      </div>
    )
  }

  // Error state (failed preview)
  if (error && !previewData) {
    return (
      <div className="max-w-screen-xl mx-auto space-y-6">
        <div>
          <h1 className="text-xl font-bold">Select Fencers</h1>
        </div>
        <div className="flex flex-col items-center justify-center h-64 gap-4">
          <p className="text-red-400 text-sm">{error}</p>
          <div className="flex items-center gap-3">
            <button
              onClick={handleRetry}
              className="px-4 py-2 bg-brand-500 hover:bg-brand-400 rounded-lg text-sm font-semibold transition-colors"
            >
              Retry
            </button>
            <button
              onClick={handleSkip}
              disabled={submitting}
              className="px-4 py-2 bg-gray-800 hover:bg-gray-700 disabled:opacity-40 rounded-lg text-sm font-medium transition-colors text-gray-300"
            >
              {submitting ? 'Starting...' : 'Skip -- Auto-detect'}
            </button>
          </div>
        </div>
      </div>
    )
  }

  // No detections state
  if (previewData && totalDetections === 0) {
    return (
      <div className="max-w-screen-xl mx-auto space-y-6">
        <div>
          <h1 className="text-xl font-bold">Select Fencers</h1>
        </div>
        <div className="flex flex-col items-center justify-center h-64 gap-4">
          <p className="text-gray-400 text-sm">No people detected in this video.</p>
          <button
            onClick={handleSkip}
            disabled={submitting}
            className="px-4 py-2 bg-gray-800 hover:bg-gray-700 disabled:opacity-40 rounded-lg text-sm font-medium transition-colors text-gray-300"
          >
            {submitting ? 'Starting...' : 'Skip -- Auto-detect'}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-screen-xl mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-bold">Select Fencers</h1>
        <p className="text-sm text-gray-400 mt-1">
          Click on the skeletons to identify yourself and your opponent.
          {totalDetections === 1 && (
            <span className="ml-1 text-yellow-400">
              One person detected -- assigned as you. Select Start Analysis or click to change.
            </span>
          )}
        </p>
      </div>

      {/* Main canvas area */}
      <div className="relative rounded-xl overflow-hidden bg-black border border-gray-800">
        {!imgLoaded && (
          <div className="flex items-center justify-center h-64 text-gray-500 text-sm">
            Loading frame...
          </div>
        )}

        {/* Background image */}
        <img
          ref={imgRef}
          src={imageUrl}
          alt={`Preview frame ${activeFrameIdx + 1}`}
          className={clsx('w-full', !imgLoaded && 'hidden')}
          onLoad={() => setImgLoaded(true)}
          draggable={false}
        />

        {/* Overlay canvas */}
        <canvas
          ref={canvasRef}
          className={clsx(
            'absolute inset-0 w-full h-full cursor-pointer',
            !imgLoaded && 'hidden',
          )}
          onClick={handleCanvasClick}
        />
      </div>

      {/* Frame thumbnails */}
      {previewData && previewData.frames.length > 1 && (
        <div className="flex items-center gap-2">
          {previewData.frames.map((frame, idx) => (
            <button
              key={frame.frame_index}
              onClick={() => setActiveFrameIdx(idx)}
              className={clsx(
                'relative w-20 h-14 rounded-lg overflow-hidden border-2 transition-colors',
                idx === activeFrameIdx
                  ? 'border-brand-500'
                  : 'border-gray-700 hover:border-gray-500',
              )}
            >
              <img
                src={`/uploads/${frame.image_key}`}
                alt={`Frame ${idx + 1}`}
                className="w-full h-full object-cover"
                draggable={false}
              />
              <span className="absolute bottom-0.5 right-1 text-[10px] font-mono text-white/80 bg-black/60 px-1 rounded">
                {idx + 1}
              </span>
            </button>
          ))}
        </div>
      )}

      {/* Selection status */}
      <div className="flex items-center gap-4 text-sm">
        <span className="text-gray-400">Selection:</span>
        <span
          className={clsx(
            'px-2.5 py-1 rounded-lg font-medium',
            fencerSelection
              ? 'bg-orange-600/20 text-orange-400 border border-orange-600/40'
              : 'bg-gray-800 text-gray-500',
          )}
        >
          Me {fencerSelection ? '\u2713' : '\u2014'}
        </span>
        <span
          className={clsx(
            'px-2.5 py-1 rounded-lg font-medium',
            opponentSelection
              ? 'bg-blue-600/20 text-blue-400 border border-blue-600/40'
              : 'bg-gray-800 text-gray-500',
          )}
        >
          Opponent {opponentSelection ? '\u2713' : '\u2014'}
        </span>
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {/* Action buttons */}
      <div className="flex items-center gap-4">
        <button
          onClick={handleSubmit}
          disabled={submitting || !fencerSelection}
          className="px-6 py-2.5 bg-brand-500 hover:bg-brand-400 disabled:opacity-40 disabled:cursor-not-allowed rounded-lg text-sm font-semibold transition-colors"
        >
          {submitting ? 'Starting...' : 'Start Analysis'}
        </button>
        <button
          onClick={handleSkip}
          disabled={submitting}
          className="px-4 py-2.5 bg-gray-800 hover:bg-gray-700 disabled:opacity-40 rounded-lg text-sm font-medium transition-colors text-gray-300"
        >
          Skip -- Auto-detect
        </button>
      </div>
    </div>
  )
}
