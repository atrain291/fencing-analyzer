import { useParams } from 'react-router-dom'
import { useEffect, useRef, useState } from 'react'
import api from '@/api/client'

interface AnalysisSummary {
  llm_summary: string
  technique_scores: Record<string, number>
}

export default function VideoReview() {
  const { boutId } = useParams<{ boutId: string }>()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [analysis, setAnalysis] = useState<AnalysisSummary | null>(null)
  const [videoUrl, setVideoUrl] = useState('')

  useEffect(() => {
    if (!boutId) return
    api.get(`/bouts/${boutId}`).then(({ data }) => {
      if (data.video_url) setVideoUrl(data.video_url)
      if (data.analysis) setAnalysis(data.analysis)
    })
  }, [boutId])

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
            {/* Canvas overlay for skeleton rendering — wired up in Stage 1 completion */}
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />
          </div>
          <p className="text-xs text-gray-500">
            Skeleton overlay renders here — canvas pipeline wired in Stage 1 completion
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
              <p className="text-sm text-gray-500">Loading analysis…</p>
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
