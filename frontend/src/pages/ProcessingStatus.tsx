import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getBoutStatus, deleteBout, type BoutStatus } from '@/api/bouts'
import { Loader2, CheckCircle, XCircle } from 'lucide-react'

const STAGE_LABELS: Record<string, string> = {
  ingest: 'Ingesting video',
  pose_estimation: 'Running pose estimation',
  action_classification: 'Classifying actions',
  llm_synthesis: 'Generating coaching feedback',
  complete: 'Complete',
}

export default function ProcessingStatus() {
  const { boutId } = useParams<{ boutId: string }>()
  const navigate = useNavigate()
  const [status, setStatus] = useState<BoutStatus | null>(null)

  useEffect(() => {
    if (!boutId) return
    const id = Number(boutId)

    const poll = async () => {
      try {
        const s = await getBoutStatus(id)
        setStatus(s)
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

  const pct = status?.pipeline_progress?.pct ?? 0
  const stage = status?.pipeline_progress?.stage ?? ''
  const stageLabel = STAGE_LABELS[stage] ?? stage

  return (
    <div className="max-w-lg mx-auto mt-20 text-center space-y-6">
      {status?.status === 'failed' ? (
        <>
          <XCircle className="mx-auto text-red-400" size={48} />
          <p className="text-red-400 font-medium">Analysis failed</p>
          <p className="text-gray-500 text-sm">{status.error}</p>
        </>
      ) : status?.status === 'complete' ? (
        <>
          <CheckCircle className="mx-auto text-green-400" size={48} />
          <p className="font-medium">Analysis complete — loading review…</p>
        </>
      ) : (
        <>
          <Loader2 className="mx-auto text-brand-500 animate-spin" size={48} />
          <p className="font-medium">{stageLabel || 'Queued…'}</p>
          <div className="bg-gray-800 rounded-full h-3 w-full">
            <div
              className="bg-brand-500 h-3 rounded-full transition-all duration-500"
              style={{ width: `${pct}%` }}
            />
          </div>
          <p className="text-gray-500 text-sm">{pct}%</p>
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
        </>
      )}
    </div>
  )
}
