import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, User, AlertCircle, FileVideo, Clock, Trash2 } from 'lucide-react'
import { listFencers, createFencer, deleteFencer, type Fencer } from '@/api/fencers'
import { uploadVideo, deleteBout, listBouts, getThumbnailUrl, type BoutSummary } from '@/api/bouts'
import clsx from 'clsx'

function formatDuration(ms: number): string {
  const totalSec = Math.floor(ms / 1000)
  const min = Math.floor(totalSec / 60)
  const sec = totalSec % 60
  return `${min}:${sec.toString().padStart(2, '0')}`
}

function BoutThumbnail({ boutId }: { boutId: number }) {
  const [failed, setFailed] = useState(false)
  if (failed) {
    return (
      <div className="w-[120px] h-[68px] rounded bg-gray-700 flex-shrink-0 flex items-center justify-center">
        <FileVideo size={24} className="text-gray-500" />
      </div>
    )
  }
  return (
    <img
      src={getThumbnailUrl(boutId)}
      alt="Bout thumbnail"
      className="w-[120px] h-[68px] rounded object-cover bg-gray-700 flex-shrink-0"
      onError={() => setFailed(true)}
    />
  )
}

function StatusBadge({ status }: { status: string }) {
  const label = status.charAt(0).toUpperCase() + status.slice(1).replace('_', ' ')
  const color =
    status === 'complete' || status === 'done'
      ? 'bg-green-500/20 text-green-400'
      : status === 'processing' || status === 'queued' || status === 'previewing'
        ? 'bg-yellow-500/20 text-yellow-400'
        : status === 'configuring' || status === 'preview_ready'
          ? 'bg-blue-500/20 text-blue-400'
          : status === 'failed'
            ? 'bg-red-500/20 text-red-400'
            : 'bg-gray-500/20 text-gray-400'
  return <span className={clsx('px-2 py-0.5 rounded-full text-xs font-medium', color)}>{label}</span>
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [fencers, setFencers] = useState<Fencer[]>([])
  const [selectedFencer, setSelectedFencer] = useState<number | null>(null)
  const [newFencerName, setNewFencerName] = useState('')
  const [uploadPct, setUploadPct] = useState<number | null>(null)
  const [error, setError] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const [bouts, setBouts] = useState<BoutSummary[]>([])
  const abortRef = useRef<AbortController | null>(null)
  const boutIdRef = useRef<number | null>(null)

  useEffect(() => {
    listFencers().then(setFencers).catch(() => setError('Could not load fencers'))
  }, [])

  useEffect(() => {
    if (selectedFencer) {
      listBouts(selectedFencer).then(setBouts).catch(() => {})
    } else {
      setBouts([])
    }
  }, [selectedFencer])

  async function handleCreateFencer() {
    const name = newFencerName.trim()
    if (!name) return
    if (fencers.some(f => f.name.toLowerCase() === name.toLowerCase())) {
      setError('A fencer with this name already exists')
      return
    }
    setError('')
    try {
      const f = await createFencer(name)
      setFencers(prev => [f, ...prev])
      setSelectedFencer(f.id)
      setNewFencerName('')
    } catch (err: unknown) {
      const msg = (err as { response?: { status?: number } })?.response?.status === 409
        ? 'A fencer with this name already exists'
        : 'Failed to create fencer profile'
      setError(msg)
    }
  }

  async function handleDeleteFencer(fencerId: number) {
    const fencer = fencers.find(f => f.id === fencerId)
    if (!fencer) return
    if (!window.confirm(`Delete "${fencer.name}" and all their bouts? This cannot be undone.`)) return
    setFencers(prev => prev.filter(f => f.id !== fencerId))
    if (selectedFencer === fencerId) {
      setSelectedFencer(null)
      setBouts([])
    }
    try {
      await deleteFencer(fencerId)
    } catch {
      listFencers().then(setFencers).catch(() => {})
      setError('Failed to delete fencer')
    }
  }

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file || !selectedFencer) return
    setError('')
    setUploadPct(0)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const res = await uploadVideo(file, selectedFencer, pct => setUploadPct(pct), controller.signal)
      boutIdRef.current = res.bout_id
      navigate(`/bouts/${res.bout_id}/configure`)
    } catch {
      setError('Upload failed. Check that the API is running.')
      setUploadPct(null)
    }
  }

  async function handleCancel() {
    abortRef.current?.abort()
    if (boutIdRef.current !== null) {
      try {
        await deleteBout(boutIdRef.current)
      } catch {
        // ignore cleanup errors
      }
      boutIdRef.current = null
    }
    setUploadPct(null)
  }

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      <h1 className="text-2xl font-bold">Fencing Analyzer</h1>

      {/* Fencer selection */}
      <section className="bg-gray-900 rounded-xl p-5 space-y-4">
        <h2 className="font-semibold flex items-center gap-2">
          <User size={16} /> Fencer Profile
        </h2>
        <div className="flex gap-2">
          <input
            className="flex-1 bg-gray-800 rounded-lg px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-brand-500"
            placeholder="New fencer name…"
            value={newFencerName}
            onChange={e => setNewFencerName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleCreateFencer()}
          />
          <button
            className="px-4 py-2 bg-brand-500 hover:bg-sky-400 rounded-lg text-sm font-medium transition-colors"
            onClick={handleCreateFencer}
          >
            Add
          </button>
        </div>
        <div className="space-y-1">
          {fencers.map(f => (
            <div key={f.id} className="flex items-center gap-1">
              <button
                onClick={() => setSelectedFencer(f.id)}
                className={clsx(
                  'flex-1 text-left px-3 py-2 rounded-lg text-sm transition-colors',
                  selectedFencer === f.id
                    ? 'bg-brand-500/20 text-brand-500 ring-1 ring-brand-500'
                    : 'hover:bg-gray-800 text-gray-300',
                )}
              >
                {f.name}
              </button>
              <button
                onClick={() => handleDeleteFencer(f.id)}
                className="flex-shrink-0 p-2 rounded-lg text-gray-500 hover:text-red-400 hover:bg-red-400/10 transition-colors"
                title="Delete fencer"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
          {fencers.length === 0 && (
            <p className="text-gray-500 text-sm">No fencer profiles yet. Add one above.</p>
          )}
        </div>
      </section>

      {/* Bout list */}
      {selectedFencer && (
        <section className="bg-gray-900 rounded-xl p-5 space-y-4">
          <h2 className="font-semibold flex items-center gap-2">
            <FileVideo size={16} /> Your Bouts
          </h2>
          {bouts.length === 0 ? (
            <p className="text-gray-500 text-sm">No bouts yet. Upload a video to get started.</p>
          ) : (
            <div className="space-y-2">
              {bouts.map(bout => (
                <button
                  key={bout.id}
                  onClick={() => {
                    const s = bout.status
                    if (s === 'complete' || s === 'done') {
                      navigate(`/bouts/${bout.id}/review`)
                    } else if (s === 'configuring' || s === 'preview_ready' || s === 'failed') {
                      navigate(`/bouts/${bout.id}/configure`)
                    } else {
                      navigate(`/bouts/${bout.id}/processing`)
                    }
                  }}
                  className="w-full flex items-center gap-4 bg-gray-800 hover:bg-gray-750 rounded-lg p-3 text-left cursor-pointer transition-colors"
                >
                  <BoutThumbnail boutId={bout.id} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm text-gray-300">
                        {new Date(bout.created_at).toLocaleDateString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          year: 'numeric',
                        })}
                      </span>
                      <StatusBadge status={bout.status} />
                    </div>
                    <div className="text-xs text-gray-500 flex items-center gap-1">
                      {bout.duration_ms ? (
                        <>
                          <Clock size={12} />
                          {formatDuration(bout.duration_ms)}
                        </>
                      ) : (
                        '\u2014'
                      )}
                    </div>
                  </div>
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={(e) => {
                      e.stopPropagation()
                      setBouts(prev => prev.filter(b => b.id !== bout.id))
                      deleteBout(bout.id).catch(() => {
                        // Re-fetch on failure to restore the list
                        if (selectedFencer) {
                          listBouts(selectedFencer).then(setBouts).catch(() => {})
                        }
                      })
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.stopPropagation()
                        e.preventDefault()
                        setBouts(prev => prev.filter(b => b.id !== bout.id))
                        deleteBout(bout.id).catch(() => {
                          if (selectedFencer) {
                            listBouts(selectedFencer).then(setBouts).catch(() => {})
                          }
                        })
                      }
                    }}
                    className="flex-shrink-0 p-2 rounded-lg text-gray-500 hover:text-red-400 hover:bg-red-400/10 transition-colors"
                    title="Delete bout"
                  >
                    <Trash2 size={16} />
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>
      )}

      {/* Upload */}
      <section className="bg-gray-900 rounded-xl p-5 space-y-4">
        <h2 className="font-semibold flex items-center gap-2">
          <Upload size={16} /> Upload Bout Video
        </h2>
        <button
          disabled={!selectedFencer || uploadPct !== null}
          onClick={() => fileRef.current?.click()}
          className={clsx(
            'w-full border-2 border-dashed rounded-xl p-10 text-center transition-colors',
            selectedFencer
              ? 'border-gray-600 hover:border-brand-500 hover:bg-brand-500/5 cursor-pointer'
              : 'border-gray-700 text-gray-600 cursor-not-allowed',
          )}
        >
          {uploadPct !== null ? (
            <div className="space-y-2">
              <p className="text-sm text-gray-400">Uploading… {uploadPct}%</p>
              <div className="bg-gray-800 rounded-full h-2">
                <div
                  className="bg-brand-500 h-2 rounded-full transition-all"
                  style={{ width: `${uploadPct}%` }}
                />
              </div>
            </div>
          ) : (
            <>
              <Upload className="mx-auto mb-2 text-gray-500" size={32} />
              <p className="text-sm text-gray-400">
                {selectedFencer ? 'Click to select a video file' : 'Select a fencer profile first'}
              </p>
              <p className="text-xs text-gray-600 mt-1">MP4, MOV, AVI, MKV, WebM</p>
            </>
          )}
        </button>
        <input ref={fileRef} type="file" accept="video/*" className="hidden" onChange={handleFileChange} />
        {uploadPct !== null && (
          <button
            onClick={handleCancel}
            className="w-full py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm font-medium transition-colors text-gray-300"
          >
            Cancel
          </button>
        )}
      </section>

      {error && (
        <p className="flex items-center gap-2 text-red-400 text-sm">
          <AlertCircle size={14} /> {error}
        </p>
      )}
    </div>
  )
}
