import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, User, AlertCircle } from 'lucide-react'
import { listFencers, createFencer, type Fencer } from '@/api/fencers'
import { uploadVideo, deleteBout } from '@/api/bouts'
import clsx from 'clsx'

export default function Dashboard() {
  const navigate = useNavigate()
  const [fencers, setFencers] = useState<Fencer[]>([])
  const [selectedFencer, setSelectedFencer] = useState<number | null>(null)
  const [newFencerName, setNewFencerName] = useState('')
  const [uploadPct, setUploadPct] = useState<number | null>(null)
  const [error, setError] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const boutIdRef = useRef<number | null>(null)

  useEffect(() => {
    listFencers().then(setFencers).catch(() => setError('Could not load fencers'))
  }, [])

  async function handleCreateFencer() {
    if (!newFencerName.trim()) return
    try {
      const f = await createFencer(newFencerName.trim())
      setFencers(prev => [f, ...prev])
      setSelectedFencer(f.id)
      setNewFencerName('')
    } catch {
      setError('Failed to create fencer profile')
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
            <button
              key={f.id}
              onClick={() => setSelectedFencer(f.id)}
              className={clsx(
                'w-full text-left px-3 py-2 rounded-lg text-sm transition-colors',
                selectedFencer === f.id
                  ? 'bg-brand-500/20 text-brand-500 ring-1 ring-brand-500'
                  : 'hover:bg-gray-800 text-gray-300',
              )}
            >
              {f.name}
            </button>
          ))}
          {fencers.length === 0 && (
            <p className="text-gray-500 text-sm">No fencer profiles yet. Add one above.</p>
          )}
        </div>
      </section>

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
