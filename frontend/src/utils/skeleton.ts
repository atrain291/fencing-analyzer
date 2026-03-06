import type { Keypoint } from '@/api/bouts'

/**
 * COCO 17-point skeleton connections for drawing pose overlays.
 * Each tuple is [startJoint, endJoint].
 */
export const SKELETON_EDGES: [string, string][] = [
  ['nose', 'left_eye'], ['nose', 'right_eye'],
  ['left_eye', 'left_ear'], ['right_eye', 'right_ear'],
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
]

/** Minimum keypoint confidence required to draw a joint or edge. */
export const CONFIDENCE_THRESHOLD = 0.3

/**
 * Draw a full skeleton (edges + keypoints) onto a canvas context.
 *
 * Coordinates in `pose` are normalised 0-1; `width` and `height` are
 * the canvas pixel dimensions used to scale them.
 */
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  pose: Record<string, Keypoint>,
  width: number,
  height: number,
  color: string,
): void {
  // Pose models may return (0,0) for undetected keypoints, sometimes with
  // non-zero confidence. Treat any keypoint at the exact origin as invalid.
  const valid = (kp: Keypoint) =>
    kp.confidence >= CONFIDENCE_THRESHOLD && (kp.x !== 0 || kp.y !== 0)

  // Draw edges
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  for (const [start, end] of SKELETON_EDGES) {
    const p1 = pose[start]
    const p2 = pose[end]
    if (!p1 || !p2) continue
    if (!valid(p1) || !valid(p2)) continue

    const alpha = Math.min(p1.confidence, p2.confidence)
    ctx.globalAlpha = alpha
    ctx.beginPath()
    ctx.moveTo(p1.x * width, p1.y * height)
    ctx.lineTo(p2.x * width, p2.y * height)
    ctx.stroke()
  }

  // Draw keypoints
  ctx.globalAlpha = 1
  for (const [_name, kp] of Object.entries(pose)) {
    if (!valid(kp)) continue

    ctx.globalAlpha = kp.confidence
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(kp.x * width, kp.y * height, 4, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.globalAlpha = 1
}
