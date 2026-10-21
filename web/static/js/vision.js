// Camera plus in-browser MediaPipe (hands, pose, face). Emits one "frame" per video frame in the same
// layout signbridge/landmarks.py expects: {lh, rh, pose, lips}. Detectors get the un-mirrored video; mirroring is CSS only.
import { config, setCameraState } from './api.js';

const BASE = '/static/vendor/tasks-vision';
export const HAND_BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const r4 = (x) => Math.round(x * 1e4) / 1e4;

/** Same rule as assign_hands in signbridge/landmarks.py: nearest pose wrist wins, a lowered hand counts as absent. */
export function assignHands(hands, pose) {
  if (pose) {
    const limit = (pose[1][1] + pose[2][1]) / 2 + Math.hypot(pose[1][0] - pose[2][0], pose[1][1] - pose[2][1]);
    hands = hands.filter((h) => h[0][1] <= limit);
  }
  hands = hands.slice(0, 2);
  if (!hands.length) return [null, null];
  const d = (h, w) => Math.hypot(h[0][0] - w[0], h[0][1] - w[1]);
  if (pose) { // pose[5] is the left wrist, pose[6] the right
    if (hands.length === 1) return d(hands[0], pose[5]) < d(hands[0], pose[6]) ? [hands[0], null] : [null, hands[0]];
    return d(hands[0], pose[5]) + d(hands[1], pose[6]) <= d(hands[0], pose[6]) + d(hands[1], pose[5]) ? [hands[0], hands[1]] : [hands[1], hands[0]];
  }
  if (hands.length === 1) return hands[0][0][0] > 0.5 ? [hands[0], null] : [null, hands[0]];
  hands.sort((a, b) => b[0][0] - a[0][0]);
  return [hands[0], hands[1]];
}

export class Vision {
  constructor(video, canvas) {
    this.video = video;
    this.canvas = canvas;
    this.listeners = new Set();
    this.running = false;
    this.n = 0;
    this.lastTs = 0;
    this.lips = null;
    this.blend = null;
    this.fps = 0;
  }

  /** Subscribe to frames. Returns an unsubscribe function. */
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }

  async start() {
    const cfg = await config();
    this.cfg = cfg;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: 'user' }, audio: false });
    } catch (e) {
      setCameraState('error');
      throw e;
    }
    this.video.srcObject = this.stream;
    await this.video.play();
    await this.#load();
    this.canvas.width = this.video.videoWidth;
    this.canvas.height = this.video.videoHeight;
    const css = getComputedStyle(document.documentElement);
    this.colors = { main: css.getPropertyValue('--sc-accent').trim(), other: css.getPropertyValue('--sc-ink-soft').trim() };
    this.running = true;
    setCameraState('on');
    this.#loop();
  }

  stop() {
    this.running = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.video.srcObject = null;
    setCameraState('off');
  }

  async #load() {
    const { FilesetResolver, HandLandmarker, PoseLandmarker, FaceLandmarker } = await import(`${BASE}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${BASE}/wasm`);
    const make = async (Cls, model, opts) => {
      const options = (delegate) => ({ baseOptions: { modelAssetPath: `/static/models/${model}`, delegate }, runningMode: 'VIDEO', ...opts });
      try { return await Cls.createFromOptions(fileset, options('GPU')); } // shortcut: falls back to CPU when WebGL is unavailable
      catch { return Cls.createFromOptions(fileset, options('CPU')); }
    };
    [this.handLm, this.poseLm, this.faceLm] = await Promise.all([
      make(HandLandmarker, 'hand_landmarker.task', { numHands: 2 }),
      make(PoseLandmarker, 'pose_landmarker_lite.task', { numPoses: 1 }),
      make(FaceLandmarker, 'face_landmarker.task', { numFaces: 1, outputFaceBlendshapes: true })]);
  }

  #loop() {
    if (!this.running) return;
    this.video.requestVideoFrameCallback(() => {
      if (!this.running) return;
      const now = performance.now();
      if (this.prev) this.fps = 0.9 * this.fps + 0.1 * (1000 / (now - this.prev)); // smoothed frames per second
      this.prev = now;
      this.#process();
      this.#loop();
    });
  }

  #process() {
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
      this.blend = cats ? { names: cats.map((c) => c.categoryName), values: cats.map((c) => c.score) } : null;
    }
    const hands = hr.landmarks.map((h) => h.map((p) => [r4(p.x), r4(p.y), r4(p.z)]));
    const lm = pr.landmarks[0];
    const pose = lm ? this.cfg.POSE_IDX.map((i) => [r4(lm[i].x), r4(lm[i].y)]) : null;
    const [lh, rh] = assignHands(hands, pose);
    const xy = (h) => h && h.map((p) => [p[0], p[1]]);
    const frame = { lh: xy(lh), rh: xy(rh), pose, lips: this.lips };
    this.#draw(lh, rh);
    const event = { frame, hand: rh ?? lh, aspect: v.videoWidth / v.videoHeight, blend: this.blend, pose };
    for (const fn of this.listeners) fn(event);
  }

  #draw(lh, rh) {
    const ctx = this.canvas.getContext('2d');
    const { width: w, height: h } = this.canvas;
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = Math.max(2, w / 320);
    ctx.lineCap = 'round';
    for (const [hand, color] of [[lh, this.colors.other], [rh, this.colors.main]]) {
      if (!hand) continue;
      ctx.strokeStyle = ctx.fillStyle = color;
      ctx.beginPath();
      for (const [a, b] of HAND_BONES) { ctx.moveTo(hand[a][0] * w, hand[a][1] * h); ctx.lineTo(hand[b][0] * w, hand[b][1] * h); }
      ctx.stroke();
      for (const p of hand) { ctx.beginPath(); ctx.arc(p[0] * w, p[1] * h, ctx.lineWidth * 1.1, 0, 7); ctx.fill(); }
    }
  }
}

/** Wire the stage's start button and its message panel (ids: stage-msg, stage-title, stage-text, start-camera). Returns start(). */
export function wireCamera(vision, { onReady } = {}) {
  const $ = (id) => document.getElementById(id);
  const [msg, title, text, button] = ['stage-msg', 'stage-title', 'stage-text', 'start-camera'].map($);
  const show = (t, x, b) => { title.textContent = t; text.textContent = x; button.lastChild.textContent = b; msg.hidden = false; };
  async function start() {
    button.disabled = true;
    show('Starting the camera…', 'Loading the hand, body and face models. This takes a few seconds.', 'Starting…');
    try {
      await vision.start();
      msg.hidden = true;
      await onReady?.();
    } catch (e) {
      const denied = e.name === 'NotAllowedError' || e.name === 'SecurityError';
      show(denied ? 'Camera access is blocked' : 'The camera did not start',
        denied ? 'Click the camera icon in the address bar, choose Allow, then try again.'
          : e.name === 'NotFoundError' ? 'No camera was found on this computer.' : e.message, 'Try again');
    } finally { button.disabled = false; }
  }
  button.addEventListener('click', start);
  navigator.permissions?.query({ name: 'camera' }).then((p) => p.state === 'granted' && start()).catch(() => {});
  return start;
}
