import type { Keypoint } from '@/api/bouts'

/**
 * Skeleton connections for drawing pose overlays.
 * COCO 17-point body + 6 foot keypoints from RTMPose WholeBody.
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
  // Foot bones (ankle -> heel, ankle -> big toe, big toe -> small toe)
  ['left_ankle', 'left_heel'], ['left_ankle', 'left_big_toe'],
  ['left_big_toe', 'left_small_toe'],
  ['right_ankle', 'right_heel'], ['right_ankle', 'right_big_toe'],
  ['right_big_toe', 'right_small_toe'],
]

/** Hand skeleton edges for both hands (wrist -> fingers with MCP line). */
function _handEdges(prefix: string): [string, string][] {
  const p = (j: string) => `${prefix}_${j}` as string
  return [
    // Wrist to finger bases
    [p('wrist'), p('thumb_cmc')], [p('wrist'), p('index_mcp')],
    [p('wrist'), p('pinky_mcp')],
    // Thumb chain
    [p('thumb_cmc'), p('thumb_mcp')], [p('thumb_mcp'), p('thumb_ip')],
    [p('thumb_ip'), p('thumb_tip')],
    // Index finger
    [p('index_mcp'), p('index_pip')], [p('index_pip'), p('index_dip')],
    [p('index_dip'), p('index_tip')],
    // Middle finger
    [p('middle_mcp'), p('middle_pip')], [p('middle_pip'), p('middle_dip')],
    [p('middle_dip'), p('middle_tip')],
    // Ring finger
    [p('ring_mcp'), p('ring_pip')], [p('ring_pip'), p('ring_dip')],
    [p('ring_dip'), p('ring_tip')],
    // Pinky
    [p('pinky_mcp'), p('pinky_pip')], [p('pinky_pip'), p('pinky_dip')],
    [p('pinky_dip'), p('pinky_tip')],
    // MCP line across knuckles
    [p('index_mcp'), p('middle_mcp')], [p('middle_mcp'), p('ring_mcp')],
    [p('ring_mcp'), p('pinky_mcp')],
    // Wrist to middle MCP (palm center line)
    [p('wrist'), p('middle_mcp')],
  ]
}

export const HAND_SKELETON_EDGES: [string, string][] = [
  ..._handEdges('lh'),
  ..._handEdges('rh'),
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
  showHands: boolean = false,
): void {
  // Pose models may return (0,0) for undetected keypoints, sometimes with
  // non-zero confidence. Treat any keypoint at the exact origin as invalid.
  const valid = (kp: Keypoint) =>
    kp.confidence >= CONFIDENCE_THRESHOLD && (kp.x !== 0 || kp.y !== 0)

  const edges = showHands
    ? [...SKELETON_EDGES, ...HAND_SKELETON_EDGES]
    : SKELETON_EDGES

  // Draw edges
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  for (const [start, end] of edges) {
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
