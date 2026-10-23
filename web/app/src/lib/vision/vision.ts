// Camera plus in-browser MediaPipe (hands, pose, face). Emits one "frame" per video frame in the same
// layout signbridge/landmarks.py expects: {lh, rh, pose, lips}. Detectors get the un-mirrored video; mirroring is CSS only.
import { loadConfig, type Cfg } from '../api';
import { getPrefs } from '../stores/prefs.svelte';
import { setCameraState } from '../stores/camera.svelte';

// The vendored 0.10.14 bundle matches the landmarks the models were trained on. Never use the npm package.
const BASE = '/static/vendor/tasks-vision';
export const HAND_BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const r4 = (x: number) => Math.round(x * 1e4) / 1e4;

export type Pt = number[];
export type Hand = Pt[];
export type Frame = { lh: Hand | null; rh: Hand | null; pose: Pt[] | null; lips: Pt[] | null };
export type Blend = { names: string[]; values: number[] };
export type VisionEvent = { frame: Frame; hand: Hand | null; aspect: number; blend: Blend | null; pose: Pt[] | null };

/** Same rule as assign_hands in signbridge/landmarks.py: nearest pose wrist wins, a lowered hand counts as absent. */
export function assignHands(hands: Hand[], pose: Pt[] | null): [Hand | null, Hand | null] {
  if (pose) {
    const limit = (pose[1][1] + pose[2][1]) / 2 + Math.hypot(pose[1][0] - pose[2][0], pose[1][1] - pose[2][1]);
    hands = hands.filter((h) => h[0][1] <= limit);
  }
  hands = hands.slice(0, 2);
  if (!hands.length) return [null, null];
  const d = (h: Hand, w: Pt) => Math.hypot(h[0][0] - w[0], h[0][1] - w[1]);
  if (pose) { // pose[5] is the left wrist, pose[6] the right
    if (hands.length === 1) return d(hands[0], pose[5]) < d(hands[0], pose[6]) ? [hands[0], null] : [null, hands[0]];
    return d(hands[0], pose[5]) + d(hands[1], pose[6]) <= d(hands[0], pose[6]) + d(hands[1], pose[5]) ? [hands[0], hands[1]] : [hands[1], hands[0]];
  }
  if (hands.length === 1) return hands[0][0][0] > 0.5 ? [hands[0], null] : [null, hands[0]];
  hands.sort((a, b) => b[0][0] - a[0][0]);
  return [hands[0], hands[1]];
}

export class Vision {
  video: HTMLVideoElement;
  canvas: HTMLCanvasElement;
  listeners = new Set<(ev: VisionEvent) => void>();
  running = false;
  fps = 0;
  private starting = false;
  private n = 0;
  private lastTs = 0;
  private prev = 0;
  private lips: Pt[] | null = null;
  private blend: Blend | null = null;
  private cfg!: Cfg;
  private stream?: MediaStream;
  private colors = { main: '', other: '' };
  private handLm: any; private poseLm: any; private faceLm: any;

  constructor(video: HTMLVideoElement, canvas: HTMLCanvasElement) {
    this.video = video;
    this.canvas = canvas;
  }

  /** Subscribe to frames. Returns an unsubscribe function. */
  on(fn: (ev: VisionEvent) => void) { this.listeners.add(fn); return () => { this.listeners.delete(fn); }; }

  async start() {
    if (this.running || this.starting) return;
    this.starting = true;
    this.cfg = await loadConfig();
    try {
      const size = { width: { ideal: 640 }, height: { ideal: 360 } }, chosen = getPrefs().camera;
      try { // the camera picked in Settings, falling back to the default one if it is unplugged
        this.stream = await navigator.mediaDevices.getUserMedia({ video: chosen ? { ...size, deviceId: { exact: chosen } } : { ...size, facingMode: 'user' }, audio: false });
      } catch (e) {
        if (!chosen || (e as Error).name !== 'OverconstrainedError') throw e;
        this.stream = await navigator.mediaDevices.getUserMedia({ video: { ...size, facingMode: 'user' }, audio: false });
      }
    } catch (e) {
      this.starting = false;
      setCameraState('error');
      throw e;
    }
    try {
      this.video.srcObject = this.stream!;
      await this.video.play();
      await this.load();
    } catch (e) { // do not leave the camera on when the models fail to load
      this.stream!.getTracks().forEach((t) => t.stop());
      this.starting = false;
      setCameraState('error');
      throw e;
    }
    this.starting = false;
    this.canvas.width = this.video.videoWidth;
    this.canvas.height = this.video.videoHeight;
    const css = getComputedStyle(document.documentElement);
    this.colors = { main: css.getPropertyValue('--sc-accent').trim(), other: css.getPropertyValue('--sc-ink-soft').trim() };
    this.running = true;
    setCameraState('on');
    this.loop();
  }

  stop() {
    this.running = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.video.srcObject = null;
    setCameraState('off');
  }

  private async load() {
    const { FilesetResolver, HandLandmarker, PoseLandmarker, FaceLandmarker } = await import(/* @vite-ignore */ `${BASE}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${BASE}/wasm`);
    const make = async (Cls: any, model: string, opts: object) => {
      const options = (delegate: string) => ({ baseOptions: { modelAssetPath: `/static/models/${model}`, delegate }, runningMode: 'VIDEO', ...opts });
      // CPU (WASM SIMD) by default: it is predictable, while the GPU delegate can silently land on software WebGL and run slower.
      // Add ?gpu=1 to the address to try the GPU on a machine that has one.
      if (new URLSearchParams(location.search).get('gpu') === '1') {
        try { return await Cls.createFromOptions(fileset, options('GPU')); } catch { /* fall through to CPU */ }
      }
      return Cls.createFromOptions(fileset, options('CPU'));
    };
    [this.handLm, this.poseLm, this.faceLm] = await Promise.all([
      make(HandLandmarker, 'hand_landmarker.task', { numHands: 2 }),
      make(PoseLandmarker, 'pose_landmarker_lite.task', { numPoses: 1 }),
      make(FaceLandmarker, 'face_landmarker.task', { numFaces: 1, outputFaceBlendshapes: true })]);
  }

  private loop() {
    if (!this.running) return;
    this.video.requestVideoFrameCallback(() => {
      if (!this.running) return;
      const now = performance.now();
      if (this.prev) this.fps = 0.9 * this.fps + 0.1 * (1000 / (now - this.prev)); // smoothed frames per second
      this.prev = now;
      this.process();
      this.loop();
    });
  }

  private process() {
    const v = this.video;
    if (v.readyState < 2) return;
    const ts = Math.max(performance.now(), this.lastTs + 1); // detectors need strictly increasing timestamps
    this.lastTs = ts;
    const hr = this.handLm.detectForVideo(v, ts);
    const pr = this.poseLm.detectForVideo(v, ts);
    if (this.n++ % 2 === 0) { // the face runs on every 2nd frame; the last lips and blendshapes are reused in between
      const fr = this.faceLm.detectForVideo(v, ts);
      const face = fr.faceLandmarks[0];
      this.lips = face ? this.cfg.LIP_IDX.map((i) => [r4(face[i].x), r4(face[i].y)]) : null;
      const cats = fr.faceBlendshapes?.[0]?.categories;
      this.blend = cats ? { names: cats.map((c: any) => c.categoryName), values: cats.map((c: any) => c.score) } : null;
    }
    const hands: Hand[] = hr.landmarks.map((h: any[]) => h.map((p) => [r4(p.x), r4(p.y), r4(p.z)]));
    const lm = pr.landmarks[0];
    const pose: Pt[] | null = lm ? this.cfg.POSE_IDX.map((i) => [r4(lm[i].x), r4(lm[i].y)]) : null;
    const [lh, rh] = assignHands(hands, pose);
    const xy = (h: Hand | null) => (h ? h.map((p) => [p[0], p[1]]) : null);
    const frame: Frame = { lh: xy(lh), rh: xy(rh), pose, lips: this.lips };
    this.draw(lh, rh);
    const event: VisionEvent = { frame, hand: rh ?? lh, aspect: v.videoWidth / v.videoHeight, blend: this.blend, pose };
    for (const fn of this.listeners) fn(event);
  }

  private draw(lh: Hand | null, rh: Hand | null) {
    const ctx = this.canvas.getContext('2d')!;
    const { width: w, height: h } = this.canvas;
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = Math.max(2, w / 320);
    ctx.lineCap = 'round';
    for (const [hand, color] of [[lh, this.colors.other], [rh, this.colors.main]] as const) {
      if (!hand) continue;
      ctx.strokeStyle = ctx.fillStyle = color;
      ctx.beginPath();
      for (const [a, b] of HAND_BONES) { ctx.moveTo(hand[a][0] * w, hand[a][1] * h); ctx.lineTo(hand[b][0] * w, hand[b][1] * h); }
      ctx.stroke();
      for (const p of hand) { ctx.beginPath(); ctx.arc(p[0] * w, p[1] * h, ctx.lineWidth * 1.1, 0, 7); ctx.fill(); }
    }
  }
}
