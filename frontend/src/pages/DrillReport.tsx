import { useParams, Link } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { getDrillReport, type DrillReport as DrillReportType } from '@/api/bouts'
import { AlertCircle, ArrowLeft } from 'lucide-react'

const ACTION_LABELS: Record<string, string> = {
  lunge: 'Lunge',
  advance: 'Advance',
  retreat: 'Retreat',
  en_garde: 'En Garde',
}

function scoreColor(score: number): string {
  if (score >= 75) return '#22c55e'  // green
  if (score >= 50) return '#eab308'  // yellow
  return '#ef4444'                   // red
}

function scoreBgClass(score: number): string {
  if (score >= 75) return 'text-green-400'
  if (score >= 50) return 'text-yellow-400'
  return 'text-red-400'
}

function ScoreGauge({ score, size = 120, label }: { score: number; size?: number; label: string }) {
  const radius = (size - 12) / 2
  const circumference = 2 * Math.PI * radius
  const filled = (score / 100) * circumference
  const color = scoreColor(score)

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="#374151"
            strokeWidth={8}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={8}
            strokeDasharray={`${filled} ${circumference - filled}`}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="font-bold text-white" style={{ fontSize: size * 0.22 }}>
            {Math.round(score)}
          </span>
        </div>
      </div>
      <span className="text-xs text-gray-400 font-medium uppercase tracking-wider">{label}</span>
    </div>
  )
}

function ScoreBar({ score, label }: { score: number; label: string }) {
  const color = scoreColor(score)
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-baseline">
        <span className="text-sm text-gray-300">{label}</span>
        <span className={`text-sm font-mono font-medium ${scoreBgClass(score)}`}>
          {Math.round(score)}
        </span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${score}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

function getRecommendations(report: DrillReportType): string[] {
  const recs: string[] = []

  if (report.rhythm_score < 50) {
    recs.push('Your rhythm is inconsistent. Try using a metronome to maintain steady timing between actions.')
  } else if (report.rhythm_score < 75) {
    recs.push('Your rhythm is decent but could be more consistent. Focus on even spacing between movements.')
  }

  if (report.tempo_score < 50) {
    recs.push('Your tempo varies significantly throughout the drill. Work on maintaining a steady pace from start to finish.')
  } else if (report.tempo_score < 75) {
    recs.push('Your tempo is fairly stable but drifts at times. Try counting cadence aloud during drills.')
  }

  // Check individual action consistency
  for (const [type, data] of Object.entries(report.action_breakdown)) {
    if (data.consistency_score < 50 && data.count >= 3) {
      const label = ACTION_LABELS[type] ?? type
      recs.push(`Your ${label.toLowerCase()} movements vary a lot in duration. Practice repeating each ${label.toLowerCase()} with the same extension and recovery.`)
    }
  }

  if (report.overall_score >= 75 && recs.length === 0) {
    recs.push('Excellent drill execution! Your movements are consistent and well-timed. Try increasing tempo gradually to challenge yourself.')
  }

  if (recs.length === 0) {
    recs.push('Keep practicing to build muscle memory and consistency.')
  }

  return recs
}

export default function DrillReport() {
  const { boutId } = useParams<{ boutId: string }>()
  const [report, setReport] = useState<DrillReportType | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!boutId) return
    setLoading(true)
    getDrillReport(Number(boutId))
      .then(setReport)
      .catch((err) => {
        const msg = err?.response?.data?.detail ?? 'Failed to load drill report'
        setError(msg)
      })
      .finally(() => setLoading(false))
  }, [boutId])

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto py-12 text-center">
        <div className="animate-pulse text-gray-400">Loading drill report...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto py-12">
        <div className="flex items-center gap-3 text-red-400 bg-red-950/30 border border-red-800 rounded-xl p-4">
          <AlertCircle size={20} />
          <div>
            <p className="font-medium">Could not generate drill report</p>
            <p className="text-sm text-red-300 mt-1">{error}</p>
          </div>
        </div>
        <Link
          to={`/bouts/${boutId}/review`}
          className="inline-flex items-center gap-2 mt-4 text-sm text-gray-400 hover:text-white transition-colors"
        >
          <ArrowLeft size={14} /> Back to Bout Review
        </Link>
      </div>
    )
  }

  if (!report) return null

  const recommendations = getRecommendations(report)
  const breakdownEntries = Object.entries(report.action_breakdown).sort(
    (a, b) => b[1].count - a[1].count,
  )

  // Average consistency across all action types
  const avgConsistency =
    breakdownEntries.length > 0
      ? breakdownEntries.reduce((sum, [, d]) => sum + d.consistency_score, 0) / breakdownEntries.length
      : 0

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <Link
            to={`/bouts/${boutId}/review`}
            className="inline-flex items-center gap-1.5 text-sm text-gray-400 hover:text-white transition-colors mb-2"
          >
            <ArrowLeft size={14} /> Back to Bout Review
          </Link>
          <h1 className="text-2xl font-bold">
            Drill Report — #{boutId}
          </h1>
          <p className="text-sm text-gray-400 mt-1">
            {report.drill_type === 'footwork' ? 'Footwork Drill' : 'Mixed Drill'} &middot;{' '}
            {report.total_actions} actions &middot;{' '}
            {(report.total_duration_ms / 1000).toFixed(1)}s total
          </p>
        </div>
      </div>

      {/* Score gauges */}
      <section className="bg-gray-900 rounded-xl p-6">
        <div className="flex flex-wrap items-start justify-center gap-8 md:gap-12">
          {/* Main overall gauge */}
          <ScoreGauge score={report.overall_score} size={160} label="Overall" />

          {/* Sub-scores */}
          <div className="flex flex-wrap gap-6 items-start">
            <ScoreGauge score={report.rhythm_score} size={100} label="Rhythm" />
            <ScoreGauge score={report.tempo_score} size={100} label="Tempo" />
            <ScoreGauge score={avgConsistency} size={100} label="Consistency" />
          </div>
        </div>
      </section>

      {/* Tempo indicator */}
      <section className="bg-gray-900 rounded-xl p-5">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Tempo</h3>
            <p className="text-3xl font-bold text-white mt-1">
              {report.tempo.toFixed(1)}{' '}
              <span className="text-base font-normal text-gray-400">actions/sec</span>
            </p>
          </div>
          <div className="text-right">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Total Actions</h3>
            <p className="text-3xl font-bold text-white mt-1">{report.total_actions}</p>
          </div>
        </div>
      </section>

      {/* Action breakdown table */}
      <section className="bg-gray-900 rounded-xl p-5 space-y-3">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Action Breakdown</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 border-b border-gray-800">
                <th className="text-left py-2 px-3 font-medium">Action</th>
                <th className="text-right py-2 px-3 font-medium">Count</th>
                <th className="text-right py-2 px-3 font-medium">Avg Duration</th>
                <th className="text-right py-2 px-3 font-medium">Consistency</th>
              </tr>
            </thead>
            <tbody>
              {breakdownEntries.map(([type, data]) => (
                <tr key={type} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                  <td className="py-2.5 px-3 text-gray-200 font-medium">
                    {ACTION_LABELS[type] ?? type}
                  </td>
                  <td className="py-2.5 px-3 text-right text-gray-300 font-mono">{data.count}</td>
                  <td className="py-2.5 px-3 text-right text-gray-300 font-mono">
                    {data.avg_duration_ms.toFixed(0)} ms
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    <span className={`font-mono font-medium ${scoreBgClass(data.consistency_score)}`}>
                      {data.consistency_score.toFixed(0)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Score details */}
      <section className="bg-gray-900 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Score Details</h3>
        <ScoreBar score={report.rhythm_score} label="Rhythm (gap consistency between actions)" />
        <ScoreBar score={report.tempo_score} label="Tempo (pace consistency across drill)" />
        <ScoreBar score={avgConsistency} label="Consistency (movement uniformity)" />
      </section>

      {/* Recommendations */}
      <section className="bg-gray-900 rounded-xl p-5 space-y-3">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Recommendations</h3>
        <ul className="space-y-2">
          {recommendations.map((rec, i) => (
            <li key={i} className="flex gap-3 text-sm text-gray-300 leading-relaxed">
              <span className="mt-0.5 flex-shrink-0 w-5 h-5 rounded-full bg-brand-500/20 text-brand-500 flex items-center justify-center text-xs font-bold">
                {i + 1}
              </span>
              {rec}
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
