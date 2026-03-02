import { useParams, useNavigate } from 'react-router-dom'
import { useEffect, useRef, useState, useCallback } from 'react'
import { configureROI, getThumbnailUrl, Bbox } from '@/api/bouts'

type DrawMode = 'fencer' | 'opponent'

interface DragState {
  startX: number
  startY: number
  currentX: number
  currentY: number
}

export default function ConfigureROI() {
  const { boutId } = useParams<{ boutId: string }>()
  const navigate = useNavigate()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const [mode, setMode] = useState<DrawMode>('fencer')
  const [fencerBox, setFencerBox] = useState<Bbox | null>(null)
  const [opponentBox, setOpponentBox] = useState<Bbox | null>(null)
  const [drag, setDrag] = useState<DragState | null>(null)
  const [imgLoaded, setImgLoaded] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  const thumbnailUrl = boutId ? getThumbnailUrl(Number(boutId)) : ''

  // Load thumbnail image
  useEffect(() => {
    if (!boutId) return
    const img = new Image()
    img.onload = () => {
      imageRef.current = img
      setImgLoaded(true)
    }
    img.src = thumbnailUrl
  }, [boutId, thumbnailUrl])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const img = imageRef.current
    if (!canvas || !img) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight

    ctx.drawImage(img, 0, 0)

    function drawBox(box: Bbox, color: string, label: string) {
      if (!canvas) return
      const x = box.x1 * canvas.width
      const y = box.y1 * canvas.height
      const w = (box.x2 - box.x1) * canvas.width
      const h = (box.y2 - box.y1) * canvas.height
      ctx!.strokeStyle = color
      ctx!.lineWidth = 3
      ctx!.strokeRect(x, y, w, h)
      ctx!.fillStyle = color
      ctx!.globalAlpha = 0.15
      ctx!.fillRect(x, y, w, h)
      ctx!.globalAlpha = 1
      ctx!.font = 'bold 20px sans-serif'
      ctx!.fillStyle = color
      ctx!.fillText(label, x + 6, y + 24)
    }

    if (fencerBox) drawBox(fencerBox, '#f97316', 'You')
    if (opponentBox) drawBox(opponentBox, '#3b82f6', 'Opponent')

    // Live drag preview
    if (drag && canvas) {
      const color = mode === 'fencer' ? '#f97316' : '#3b82f6'
      const x1 = Math.min(drag.startX, drag.currentX)
      const y1 = Math.min(drag.startY, drag.currentY)
      const w = Math.abs(drag.currentX - drag.startX)
      const h = Math.abs(drag.currentY - drag.startY)
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.setLineDash([6, 3])
      ctx.strokeRect(x1, y1, w, h)
      ctx.setLineDash([])
    }
  }, [fencerBox, opponentBox, drag, mode])

  useEffect(() => {
    if (imgLoaded) draw()
  }, [imgLoaded, draw])

  function canvasCoords(e: React.MouseEvent<HTMLCanvasElement>): { x: number; y: number } {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    }
  }

  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    const { x, y } = canvasCoords(e)
    setDrag({ startX: x, startY: y, currentX: x, currentY: y })
  }

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!drag) return
    const { x, y } = canvasCoords(e)
    setDrag(d => d ? { ...d, currentX: x, currentY: y } : null)
    draw()
  }

  function handleMouseUp(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!drag) return
    const canvas = canvasRef.current!
    const { x, y } = canvasCoords(e)
    const x1 = Math.min(drag.startX, x) / canvas.width
    const y1 = Math.min(drag.startY, y) / canvas.height
    const x2 = Math.max(drag.startX, x) / canvas.width
    const y2 = Math.max(drag.startY, y) / canvas.height

    // Ignore tiny boxes (accidental clicks)
    if (x2 - x1 > 0.02 && y2 - y1 > 0.02) {
      const box: Bbox = { x1, y1, x2, y2 }
      if (mode === 'fencer') {
        setFencerBox(box)
        setMode('opponent') // auto-switch to opponent after drawing fencer
      } else {
        setOpponentBox(box)
      }
    }
    setDrag(null)
  }

  async function handleStart(skipROI = false) {
    if (!boutId) return
    setSubmitting(true)
    try {
      await configureROI(
        Number(boutId),
        skipROI ? null : fencerBox,
        skipROI ? null : opponentBox,
      )
      navigate(`/bouts/${boutId}/processing`)
    } catch (err) {
      console.error(err)
      setSubmitting(false)
    }
  }

  return (
    <div className="max-w-screen-xl mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-bold">Configure Analysis — Bout #{boutId}</h1>
        <p className="text-sm text-gray-400 mt-1">
          Draw a box around each fencer so the AI knows who to track. This fixes tracking in
          tournament footage with spectators and officials in the background.
        </p>
      </div>

      {/* Mode toggle */}
      <div className="flex items-center gap-3">
        <span className="text-sm text-gray-400">Drawing:</span>
        <button
          onClick={() => setMode('fencer')}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            mode === 'fencer'
              ? 'bg-orange-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          You {fencerBox ? '✓' : ''}
        </button>
        <button
          onClick={() => setMode('opponent')}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            mode === 'opponent'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Opponent {opponentBox ? '✓' : ''}
        </button>
        {(fencerBox || opponentBox) && (
          <button
            onClick={() => { setFencerBox(null); setOpponentBox(null) }}
            className="px-3 py-1.5 rounded-lg text-sm text-gray-500 hover:text-gray-300 transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      {/* Canvas */}
      <div className="relative rounded-xl overflow-hidden bg-black border border-gray-800">
        {!imgLoaded && (
          <div className="flex items-center justify-center h-64 text-gray-500 text-sm">
            Loading thumbnail…
          </div>
        )}
        <canvas
          ref={canvasRef}
          className={`w-full cursor-crosshair ${imgLoaded ? '' : 'hidden'}`}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => { if (drag) { setDrag(null); draw() } }}
        />
      </div>

      <p className="text-xs text-gray-500">
        Drag to draw a box around each fencer. The box only needs to cover roughly where they
        start — the tracker will follow them from there. You can skip this step if there are only
        two people in the frame.
      </p>

      {/* Action buttons */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => handleStart(false)}
          disabled={submitting || (!fencerBox && !opponentBox)}
          className="px-6 py-2.5 bg-brand-500 hover:bg-brand-400 disabled:opacity-40 disabled:cursor-not-allowed rounded-lg text-sm font-semibold transition-colors"
        >
          {submitting ? 'Starting…' : 'Start Analysis'}
        </button>
        <button
          onClick={() => handleStart(true)}
          disabled={submitting}
          className="px-4 py-2.5 bg-gray-800 hover:bg-gray-700 disabled:opacity-40 rounded-lg text-sm font-medium transition-colors text-gray-300"
        >
          Skip ROI — Start Anyway
        </button>
      </div>
    </div>
  )
}
