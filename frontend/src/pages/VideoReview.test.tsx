// @vitest-environment jsdom
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Bout, Frame, Keypoint } from '@/api/bouts'
import { getBout } from '@/api/bouts'
import VideoReview, { samplePose } from './VideoReview'

vi.mock('@/api/bouts', () => ({ getBout: vi.fn(), deleteBout: vi.fn() }))

const point = (x: number): Keypoint => ({ x, y: 0.5, z: 0.25, confidence: 1 })
const frame = (timestamp_ms: number, pose: Record<string, Keypoint>): Frame => ({
  id: timestamp_ms + 1,
  timestamp_ms,
  fencer_pose: pose,
  opponent_pose: null,
})

function bout(frames: Frame[]): Bout {
  return {
    id: 1, session_id: 1, status: 'completed', result: null,
    video_url: '/uploads/test.mp4', duration_ms: 1000, pipeline_progress: {},
    created_at: '2026-09-23T00:00:00Z', frames,
  }
}

function mount(frames: Frame[]) {
  vi.mocked(getBout).mockResolvedValue(bout(frames))
  return mountPage()
}

function mountPage() {
  const view = render(
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }} initialEntries={['/bouts/1']}>
      <Routes><Route path="/bouts/:boutId" element={<VideoReview />} /></Routes>
    </MemoryRouter>,
  )
  const video = view.container.querySelector('video')!
  const canvas = view.container.querySelector('canvas')!
  return { ...view, video, canvas }
}

let currentTime = 0
let boxWidth = 1000
let boxHeight = 1000
let resizeCallback: ResizeObserverCallback | null = null
let ctx: Record<string, ReturnType<typeof vi.fn>>

beforeEach(() => {
  currentTime = 0
  boxWidth = 1000
  boxHeight = 1000
  resizeCallback = null
  ctx = Object.fromEntries(
    ['clearRect', 'beginPath', 'moveTo', 'lineTo', 'stroke', 'arc', 'fill'].map(name => [name, vi.fn()]),
  )
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctx as unknown as CanvasRenderingContext2D)
  Object.defineProperties(HTMLVideoElement.prototype, {
    videoWidth: { configurable: true, get: () => 1920 },
    videoHeight: { configurable: true, get: () => 1080 },
    clientWidth: { configurable: true, get: () => boxWidth },
    clientHeight: { configurable: true, get: () => boxHeight },
    currentTime: { configurable: true, get: () => currentTime, set: value => { currentTime = value } },
    requestVideoFrameCallback: { configurable: true, value: undefined },
    cancelVideoFrameCallback: { configurable: true, value: undefined },
  })
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    const width = boxWidth
    const height = boxHeight
    return { x: 0, y: 0, width, height, top: 0, left: 0, bottom: height, right: width, toJSON: () => ({}) }
  })
  globalThis.ResizeObserver = class {
    constructor(callback: ResizeObserverCallback) { resizeCallback = callback }
    observe() { /* exercise initial render and media events explicitly */ }
    unobserve() {}
    disconnect() {}
  }
})

afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('VideoReview overlay', () => {
  it('interpolates depth for shared joints without inventing absent joints', () => {
    const a = frame(0, { nose: { ...point(0.2), z: 0.1 }, left_wrist: point(0.1) })
    const b = frame(100, { nose: { ...point(0.8), z: 0.7 } })
    expect(samplePose([a, b], 50, 'fencer_pose')).toEqual({
      nose: { x: 0.5, y: 0.5, z: 0.4, confidence: 1 },
    })
  })

  it('aligns a 1920x1080 video inside a 1000x1000 display box', async () => {
    const { video, canvas } = mount([frame(0, { nose: point(0.5) })])
    await waitFor(() => expect(getBout).toHaveBeenCalled())
    await act(async () => { video.dispatchEvent(new Event('loadedmetadata')) })
    expect(parseFloat(canvas.style.width)).toBeCloseTo(1000)
    expect(parseFloat(canvas.style.height)).toBeCloseTo(562.5)
    expect(parseFloat(canvas.style.top)).toBeCloseTo(218.75)
    expect(parseFloat(canvas.style.left)).toBeCloseTo(0)
    boxWidth = 1600
    boxHeight = 900
    await act(async () => { resizeCallback?.([], {} as ResizeObserver) })
    expect(parseFloat(canvas.style.width)).toBeCloseTo(1600)
    expect(parseFloat(canvas.style.height)).toBeCloseTo(900)
    expect(parseFloat(canvas.style.top)).toBeCloseTo(0)
  })

  it('does not hold a pose through a missing-pose frame', async () => {
    const { video, getByText } = mount([frame(0, { nose: point(0.25) }), frame(40, {})])
    await waitFor(() => expect(getByText('Skeleton overlay active — 2 frames loaded')).toBeTruthy())
    ctx.arc.mockClear()
    currentTime = 0.02
    await act(async () => { video.dispatchEvent(new Event('seeked')) })
    expect(ctx.arc).not.toHaveBeenCalled()
  })

  it('does not bridge gaps over 100 ms or extrapolate outside the sample window', async () => {
    const { video, getByText } = mount([frame(100, { nose: point(0.25) }), frame(300, { nose: point(0.75) })])
    await waitFor(() => expect(getByText('Skeleton overlay active — 2 frames loaded')).toBeTruthy())
    ctx.clearRect.mockClear()
    for (const seconds of [0, 0.2, 0.4]) {
      currentTime = seconds
      await act(async () => { video.dispatchEvent(new Event('seeked')) })
    }
    expect(ctx.arc).not.toHaveBeenCalled()
    expect(ctx.clearRect).toHaveBeenCalledTimes(3)
  })

  it('uses decoded media time and cancels the single pending video callback on pause', async () => {
    const callbacks = new Map<number, VideoFrameRequestCallback>()
    let nextId = 1
    const request = vi.fn((callback: VideoFrameRequestCallback) => {
      const id = nextId++
      callbacks.set(id, callback)
      return id
    })
    const cancel = vi.fn((id: number) => callbacks.delete(id))
    Object.defineProperties(HTMLVideoElement.prototype, {
      requestVideoFrameCallback: { configurable: true, value: request },
      cancelVideoFrameCallback: { configurable: true, value: cancel },
    })
    const { video, getByText, unmount } = mount([frame(0, { nose: point(0.2) }), frame(100, { nose: point(0.8) })])
    await waitFor(() => expect(getByText('Skeleton overlay active — 2 frames loaded')).toBeTruthy())
    await act(async () => {
      video.dispatchEvent(new Event('play'))
      video.dispatchEvent(new Event('play'))
    })
    expect(request).toHaveBeenCalledTimes(1)
    const [id, callback] = [...callbacks][0]
    callbacks.delete(id)
    await act(async () => callback(0, { mediaTime: 0.05 } as VideoFrameCallbackMetadata))
    expect(ctx.arc).toHaveBeenCalledWith(960, 540, 4, 0, Math.PI * 2)
    expect(request).toHaveBeenCalledTimes(2)
    await act(async () => { video.dispatchEvent(new Event('pause')) })
    expect(cancel).toHaveBeenCalledWith(2)
    expect(callbacks.size).toBe(0)
    ctx.arc.mockClear()
    currentTime = 0.05
    await act(async () => { video.dispatchEvent(new Event('seeked')) })
    expect(ctx.arc).toHaveBeenCalledWith(960, 540, 4, 0, Math.PI * 2)
    await act(async () => { video.dispatchEvent(new Event('play')) })
    unmount()
    expect(cancel).toHaveBeenCalledWith(3)
  })

  it('repaints once pose data arrives after the video is paused', async () => {
    let resolveBout!: (value: Bout) => void
    vi.mocked(getBout).mockImplementation(() => new Promise(resolve => { resolveBout = resolve }))
    const { video, getByText } = mountPage()
    await act(async () => { video.dispatchEvent(new Event('loadedmetadata')) })
    expect(ctx.clearRect).toHaveBeenCalled()
    expect(ctx.arc).not.toHaveBeenCalled()
    await act(async () => { resolveBout(bout([frame(0, { nose: point(0.5) })])) })
    expect(getByText('Skeleton overlay active — 1 frames loaded')).toBeTruthy()
    expect(ctx.arc).toHaveBeenCalledWith(960, 540, 4, 0, Math.PI * 2)
  })

  it('uses one cancellable RAF loop when decoded-frame callbacks are unavailable', async () => {
    const callbacks = new Map<number, FrameRequestCallback>()
    let nextId = 1
    const request = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      const id = nextId++
      callbacks.set(id, callback)
      return id
    })
    const cancel = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(id => { callbacks.delete(id) })
    const { video, getByText } = mount([frame(0, { nose: point(0.2) }), frame(100, { nose: point(0.8) })])
    await waitFor(() => expect(getByText('Skeleton overlay active — 2 frames loaded')).toBeTruthy())
    await act(async () => {
      video.dispatchEvent(new Event('play'))
      video.dispatchEvent(new Event('play'))
    })
    expect(request).toHaveBeenCalledTimes(1)
    currentTime = 0.05
    const [id, callback] = [...callbacks][0]
    callbacks.delete(id)
    ctx.arc.mockClear()
    await act(async () => callback(0))
    expect(ctx.arc).toHaveBeenCalledWith(960, 540, 4, 0, Math.PI * 2)
    expect(request).toHaveBeenCalledTimes(2)
    await act(async () => { video.dispatchEvent(new Event('ended')) })
    expect(cancel).toHaveBeenCalledWith(2)
    expect(callbacks.size).toBe(0)
  })
})
