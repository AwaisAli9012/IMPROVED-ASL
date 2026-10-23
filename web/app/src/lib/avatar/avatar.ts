// A 2D signing figure drawn from recorded landmark animations (/static/signs/<label>.json).
// Word clips are in shoulder-width units centred on the shoulders; letter and digit clips are a single hand
// centred on the wrist. Spelled words play as a run of letter close-ups.
import { getPrefs } from '../stores/prefs.svelte';
import { HAND_BONES, type Frame } from '../vision/vision';

const BLOCKS = ['lh', 'rh', 'pose', 'lips'] as const;
type Clip = { label: string; display?: string; kind: 'word' | 'hand'; fps?: number; frames: Frame[] };
export type Token = { type: 'sign'; label: string } | { type: 'spell'; text: string };
type Seg = { type: 'clip'; t0: number; dur: number; clip: Clip; idx: number } | { type: 'blend'; t0: number; dur: number; a: Frame; b: Frame; hand: boolean };
const BLEND_MS = 150, HOLD_MS = 600;
const BODY = '#243029';
const REST: Frame = { // a neutral standing pose, so the canvas is never empty
  lh: null, rh: null, lips: null,
  pose: [[0, -0.95], [0.5, 0], [-0.5, 0], [0.62, 0.9], [-0.62, 0.9], [0.5, 1.55], [-0.5, 1.55]],
};
const cache = new Map<string, Promise<Clip | null>>();
const loadSign = (label: string): Promise<Clip | null> => {
  if (!cache.has(label)) cache.set(label, fetch(`/static/signs/${encodeURIComponent(label)}.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null));
  return cache.get(label)!;
};

/** Tokens ({type:'sign',label} or {type:'spell',text}) to clips; a sign with no animation is spelled instead. */
async function clipsFor(tokens: Token[]) {
  const clips: Clip[] = [];
  for (const t of tokens) {
    const sign = t.type === 'sign' ? await loadSign(t.label) : null;
    if (sign) { clips.push(sign); continue; }
    for (const ch of (t.type === 'sign' ? t.label : t.text).toUpperCase()) {
      const letter = await loadSign(ch);
      if (letter) clips.push(letter);
    }
  }
  return clips;
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
function lerpFrame(a: Frame, b: Frame, t: number): Frame {
  const out: any = {};
  for (const k of BLOCKS) {
    const [p, q] = [a[k], b[k]];
    out[k] = p && q && p.length === q.length
      ? p.map((pt: number[], i: number) => (pt && q[i] ? [lerp(pt[0], q[i][0], t), lerp(pt[1], q[i][1], t)] : pt ?? q[i]))
      : (t < 0.5 ? p : q) ?? null; // a hand that appears or vanishes switches at the midpoint
  }
  return out;
}

export class Avatar {
  canvas: HTMLCanvasElement;
  mirror: boolean;
  speed: number;
  busy = false; // true while a sign is playing
  private token = 0;
  private visible = true;
  private colors: { main: string; soft: string; ink: string };
  private dead = false;
  private observers: { disconnect(): void }[] = [];

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const prefs = getPrefs();
    this.mirror = prefs.mirror; // true shows signs as you would make them, like a mirror
    this.speed = prefs.avatarSpeed;
    const css = getComputedStyle(document.documentElement);
    this.colors = { main: css.getPropertyValue('--sc-accent').trim(), soft: css.getPropertyValue('--sc-ink-soft').trim(), ink: css.getPropertyValue('--sc-ink').trim() };
    const resize = new ResizeObserver(() => { this.#fit(); this.#drawRest(); });
    const seen = new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; });
    resize.observe(canvas);
    seen.observe(canvas);
    this.observers = [resize, seen];
    this.#fit();
    this.#drawRest();
    this.#idle();
  }

  /** A slow breathing sway while nothing is signing, so the figure never looks frozen. */
  #idle() {
    const calm = () => matchMedia('(prefers-reduced-motion: reduce)').matches || document.documentElement.dataset.motion === 'reduced';
    const tick = (now: number) => {
      if (this.dead) return;
      if (!this.busy && this.visible && !calm()) {
        const s = Math.sin(now / 1100);
        this.#draw({ ...REST, pose: REST.pose!.map(([x, y], i) => [x + (i === 0 ? s * 0.012 : 0), y + s * (i === 0 ? 0.02 : 0.01)]) }, false);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  /** Stop drawing and release the observers when the page goes away. */
  destroy() { this.dead = true; this.token++; this.observers.forEach((o) => o.disconnect()); }

  setMirror(on: boolean) { this.mirror = on; }
  setSpeed(s: number) { this.speed = s; }

  /** Play tokens. Resolves when finished or stopped. onWord(caption, index) fires as each clip starts. */
  async play(tokens: Token[], { loop = false, onWord }: { loop?: boolean; onWord?: (caption: string, index: number) => void } = {}): Promise<boolean> {
    this.stop();
    const id = this.token;
    const clips = await clipsFor(tokens);
    if (id !== this.token || !clips.length) return false;
    const plan = this.#plan(clips);
    this.busy = true;
    return new Promise<boolean>((resolve) => {
      let t0: number | null = null, lastIdx = -1;
      const tick = (now: number) => {
        if (id !== this.token) return resolve(false);
        t0 ??= now;
        let t = now - t0;
        if (t >= plan.total) {
          if (!loop) { this.busy = false; this.#drawRest(); return resolve(true); }
          t0 = now; t = 0; lastIdx = -1;
        }
        const seg = (plan.segs.find((s) => t < s.t0 + s.dur) ?? plan.segs.at(-1))!;
        const f = (t - seg.t0) / seg.dur;
        if (seg.type === 'clip') {
          if (seg.idx !== lastIdx) { lastIdx = seg.idx; onWord?.(seg.clip.display ?? seg.clip.label, seg.idx); }
          this.#draw(seg.clip.frames[Math.min(Math.floor(f * seg.clip.frames.length), seg.clip.frames.length - 1)], seg.clip.kind === 'hand');
        } else this.#draw(lerpFrame(seg.a, seg.b, f), seg.hand);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  }

  stop() { this.token++; this.busy = false; this.#drawRest(); }

  /** Freeze on the first frame of one sign (used to step through a fingerspelled name). */
  async show(label: string) {
    const id = ++this.token;
    this.busy = true;
    const sign = await loadSign(label);
    if (sign && id === this.token) this.#draw(sign.frames[0], sign.kind === 'hand');
  }

  #plan(clips: Clip[]) {
    const segs: Seg[] = [];
    let t = 0, prev: { frame: Frame; hand: boolean } | null = null;
    clips.forEach((c, idx) => {
      const hand = c.kind === 'hand';
      const frames = c.frames.length;
      const dur = (hand ? Math.max(HOLD_MS, (frames / 24) * 1000) : (frames / (c.fps || 24)) * 1000) / this.speed;
      if (prev && prev.hand === hand) { segs.push({ type: 'blend', t0: t, dur: BLEND_MS, a: prev.frame, b: c.frames[0], hand }); t += BLEND_MS; }
      segs.push({ type: 'clip', t0: t, dur, clip: c, idx });
      t += dur;
      prev = { frame: c.frames[frames - 1], hand };
    });
    return { segs, total: t };
  }

  #fit() {
    const r = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(this.canvas.clientWidth * r));
    this.canvas.height = Math.max(1, Math.round(this.canvas.clientHeight * r));
  }

  #drawRest() { this.#draw(REST, false); }

  #draw(frame: Frame, handKind: boolean) {
    const ctx = this.canvas.getContext('2d')!;
    const { width: W, height: H } = this.canvas;
    ctx.clearRect(0, 0, W, H);
    const flip = this.mirror ? -1 : 1;
    const line = (a: number[] | null | undefined, b: number[] | null | undefined, w: number, color: string) => {
      if (!a || !b) return;
      ctx.strokeStyle = color; ctx.lineWidth = w; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    };
    const drawHand = (hand: number[][] | null, P: (p: number[]) => [number, number], color: string, w: number) => {
      if (!hand) return;
      for (const [a, b] of HAND_BONES) if (hand[a] && hand[b]) line(P(hand[a]), P(hand[b]), w, color);
      ctx.fillStyle = color;
      for (const p of hand) if (p) { ctx.beginPath(); ctx.arc(...P(p), w * 0.9, 0, 7); ctx.fill(); }
    };
    if (handKind) { // letter or digit close-up: one big hand, wrist at the bottom centre
      const u = H * 0.3;
      drawHand(frame.rh, (p): [number, number] => [W / 2 + p[0] * u * flip, H * 0.82 + p[1] * u], this.colors.main, Math.max(3, H / 70));
      return;
    }
    const u = W * 0.17;
    const P = (p: number[]): [number, number] => [W / 2 + p[0] * u * flip, H * 0.5 + p[1] * u];
    const pose = frame.pose;
    if (pose) {
      const [nose, ls, rs, le, re, lw, rw] = pose;
      if (ls && rs) { // torso
        ctx.fillStyle = BODY;
        ctx.beginPath();
        ctx.moveTo(...P([ls[0] + 0.1, ls[1] - 0.05])); ctx.lineTo(...P([rs[0] - 0.1, rs[1] - 0.05]));
        ctx.lineTo(...P([rs[0] - 0.25, 2.6])); ctx.lineTo(...P([ls[0] + 0.25, 2.6]));
        ctx.closePath(); ctx.fill();
      }
      if (nose && ls && rs) { // neck and head
        line(P(nose), P([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2]), u * 0.2, BODY);
        ctx.fillStyle = BODY;
        ctx.beginPath(); ctx.arc(...P([nose[0], nose[1] - 0.08]), u * 0.42, 0, 7); ctx.fill();
      }
      for (const [s, e, w] of [[ls, le, lw], [rs, re, rw]]) { line(P(s ?? [0, 0]), e && P(e), u * 0.16, BODY); line(e && P(e), w && P(w), u * 0.14, BODY); }
    }
    if (frame.lips) { // outer lip ring, then the inner ring: upper lip, right corner, lower lip back
      const ring = (idx: number[]) => {
        ctx.beginPath();
        idx.forEach((i: number, n: number) => { const [x, y] = P(frame.lips![i]); n ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        ctx.closePath(); ctx.stroke();
      };
      ctx.strokeStyle = this.colors.ink; ctx.lineWidth = Math.max(1.5, u * 0.02); ctx.lineJoin = 'round';
      ring([...Array(20).keys()]);
      ring([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]);
    }
    drawHand(frame.lh, P, this.colors.soft, Math.max(2, u * 0.02));
    drawHand(frame.rh, P, this.colors.main, Math.max(2, u * 0.022));
  }
}
