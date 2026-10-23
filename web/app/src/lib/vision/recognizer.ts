// Turns the stream of vision frames into signs, spelled words, tone and question markers.
// Every threshold comes from /api/config (signbridge/config.py); nothing is hard-coded here.
import { postJSON, type ApiError, type Cfg } from '../api';
import type { Blend, Frame, Hand, VisionEvent } from './vision';
import type { Vision } from './vision';

const BASELINE_KEY = 'sb.baseline';
export const loadBaseline = (): Blend | null => { try { return JSON.parse(localStorage.getItem(BASELINE_KEY) ?? 'null'); } catch { return null; } };
const saveBaseline = (b: Blend) => { try { localStorage.setItem(BASELINE_KEY, JSON.stringify(b)); } catch { /* private mode: baseline lasts for the page only */ } };

/** Average the user's neutral face over `ms` and keep it as the baseline. */
export function calibrateNeutral(vision: Vision, ms = 2000): Promise<Blend | null> {
  return new Promise((resolve) => {
    const sum: number[] = [];
    let n = 0, names: string[] = [];
    const stop = vision.on(({ blend }) => {
      if (!blend) return;
      names = blend.names;
      blend.values.forEach((v, i) => { sum[i] = (sum[i] ?? 0) + v; });
      n++;
    });
    setTimeout(() => {
      stop();
      const baseline = n ? { names, values: sum.map((s) => s / n) } : null;
      if (baseline) saveBaseline(baseline);
      resolve(baseline);
    }, ms);
  });
}

const mean = (b: Blend, keys: string[]) => keys.reduce((s, k) => s + b.values[b.names.indexOf(k)], 0) / keys.length;
const RAISE = ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'];
const DOWN = ['browDownLeft', 'browDownRight'];

export type Question = 'none' | 'yesno' | 'wh';
export type Guess = { label: string; display: string; prob: number };
export type Gloss = { type: 'sign' | 'spell'; label: string; display: string; question: Question };
export type Candidate = { display: string; prob: number; accept: boolean };
export type Result = { kind: 'word' | 'static'; top: Guess[]; accept: boolean; ms: number; frames?: number };
export type Handlers = {
  onGloss: (g: Gloss) => void; onSuggest?: (top: Guess[], question: Question) => void; onCandidate?: (c: Candidate | null) => void;
  onBusy?: (busy: boolean) => void; onChar?: (c: string, chars: string) => void; onResult?: (r: Result) => void; onError?: (e: ApiError) => void;
};
type Mode = 'words' | 'letters' | 'digits';

export class Recognizer {
  mode: Mode = 'words';
  baseline = loadBaseline();
  private cfg: Pick<Cfg, 'SEG_START_FRAMES' | 'SEG_END_FRAMES' | 'SEG_MIN_FRAMES' | 'SEG_MAX_FRAMES' | 'STATIC_STREAK' | 'SPELL_GAP_FRAMES' | 'BROW_RAISE_DELTA' | 'BROW_DOWN_DELTA'>;
  private h: Handlers;
  private pending = 0;
  private chain: Promise<void> = Promise.resolve();
  private state: 'idle' | 'capturing' = 'idle';
  private recent: Frame[] = [];
  private seg: Frame[] = [];
  private gap = 0;
  private run = 0;
  private chars = '';
  private noHand = 0;
  private streak = 0;
  private top: string | null = null;
  private committed: string | null = null;
  private inflight = false;
  private brow = { raise: 0, down: 0, n: 0 };

  constructor(cfg: Recognizer['cfg'], handlers: Handlers) {
    this.cfg = cfg;
    this.h = handlers;
    this.setMode('words');
  }

  setMode(mode: Mode) {
    this.mode = mode;
    Object.assign(this, { state: 'idle', recent: [], seg: [], gap: 0, run: 0, chars: '', noHand: 0, streak: 0, top: null, committed: null, inflight: false });
    this.brow = { raise: 0, down: 0, n: 0 };
    this.h.onCandidate?.(null);
  }

  push(ev: VisionEvent) { return this.mode === 'words' ? this.words(ev) : this.statics(ev); }

  // ---- words: a sign starts when a hand appears and ends when hands rest ------------------------
  private words({ frame, blend }: VisionEvent) {
    const c = this.cfg;
    const hasHand = !!(frame.lh || frame.rh);
    if (this.state === 'idle') {
      this.recent.push(frame);
      if (this.recent.length > c.SEG_START_FRAMES) this.recent.shift();
      this.run = hasHand ? this.run + 1 : 0;
      if (this.run >= c.SEG_START_FRAMES) {
        this.state = 'capturing';
        this.seg = [...this.recent];
        this.gap = 0;
        this.brow = { raise: 0, down: 0, n: 0 };
        this.h.onCandidate?.({ display: '…', prob: 0, accept: false });
      }
      return;
    }
    this.seg.push(frame);
    this.brows(blend);
    this.gap = hasHand ? 0 : this.gap + 1;
    if (this.gap >= c.SEG_END_FRAMES || this.seg.length >= c.SEG_MAX_FRAMES) this.endSegment();
  }

  private brows(blend: Blend | null) {
    if (!blend || !this.baseline) return; // without a calibrated neutral face there are no grammar markers
    const raise = mean(blend, RAISE) - mean(this.baseline, RAISE);
    const down = mean(blend, DOWN) - mean(this.baseline, DOWN);
    this.brow.n++;
    if (raise > this.cfg.BROW_RAISE_DELTA) this.brow.raise++;
    if (down > this.cfg.BROW_DOWN_DELTA) this.brow.down++;
  }

  private question(): Question {
    const { raise, down, n } = this.brow;
    return !n ? 'none' : down / n >= 0.3 ? 'wh' : raise / n >= 0.3 ? 'yesno' : 'none';
  }

  /** Finish the current sign now (also bound to the Space key). */
  endSegment() {
    if (this.state !== 'capturing') return;
    const seg = this.seg;
    const question = this.question();
    Object.assign(this, { state: 'idle', seg: [], recent: [], run: 0 });
    this.h.onCandidate?.(null);
    if (seg.filter((f) => f.lh || f.rh).length < this.cfg.SEG_MIN_FRAMES) return; // too short to be a sign
    this.h.onBusy?.(++this.pending > 0);
    this.chain = this.chain.then(async () => { // one request at a time keeps results in signing order
      try {
        const t0 = performance.now();
        const r = await postJSON<{ top: Guess[]; accept: boolean }>('/api/predict/word', { frames: seg }, { quiet: true });
        this.h.onResult?.({ kind: 'word', top: r.top, accept: r.accept, ms: performance.now() - t0, frames: seg.length });
        if (r.accept) this.h.onGloss({ type: 'sign', label: r.top[0].label, display: r.top[0].display, question });
        else this.h.onSuggest?.(r.top.slice(0, 3), question);
      } catch (e) { this.h.onError?.(e as ApiError); } finally { this.h.onBusy?.(--this.pending > 0); }
    });
  }

  // ---- ABC and 123: latch a character after STATIC_STREAK confident frames in a row ----------------
  private statics({ hand, aspect }: { hand: Hand | null; aspect: number }) {
    const c = this.cfg;
    if (!hand) {
      this.noHand++;
      if (this.noHand === 5) { this.committed = null; this.streak = 0; this.top = null; this.h.onCandidate?.(null); }
      if (this.noHand >= c.SPELL_GAP_FRAMES && this.chars) {
        const word = this.chars;
        this.chars = '';
        this.h.onGloss({ type: 'spell', label: `fs:${word}`, display: word.split('').join('-'), question: 'none' });
      }
      return;
    }
    this.noHand = 0;
    if (this.inflight) return; // at most one request in flight
    this.inflight = true;
    const t0 = performance.now();
    postJSON<{ top: Guess[]; accept: boolean }>('/api/predict/static', { hand, aspect, mode: this.mode }, { quiet: true }).then((r) => {
      const t = r.top[0];
      this.h.onResult?.({ kind: 'static', top: r.top, accept: r.accept, ms: performance.now() - t0 });
      this.h.onCandidate?.({ display: t.display, prob: t.prob, accept: r.accept });
      this.streak = r.accept ? (t.label === this.top ? this.streak + 1 : 1) : 0;
      this.top = r.accept ? t.label : null;
      if (this.streak >= c.STATIC_STREAK && t.label !== this.committed) {
        this.committed = t.label;
        this.chars += t.label;
        this.streak = 0;
        this.h.onChar?.(t.label, this.chars);
      }
    }).catch((e) => this.h.onError?.(e)).finally(() => { this.inflight = false; });
  }
}

/** Emotion over time: server probabilities, EMA smoothing, and a switch only after a clear, sustained change. */
export class ToneTracker {
  current = 'neutral';
  private cfg: Pick<Cfg, 'EMOTION_INTERVAL_MS' | 'EMOTION_EMA'>;
  private onTone: (name: string, ema: Record<string, number>) => void;
  private ema: Record<string, number> | null = null;
  private challenger: string | null = null;
  private count = 0;
  private last = 0;
  private inflight = false;
  private tally: Record<string, number> = {};

  constructor(cfg: ToneTracker['cfg'], onTone: ToneTracker['onTone']) {
    this.cfg = cfg;
    this.onTone = onTone;
  }

  push(blend: Blend | null, now = performance.now()) {
    if (!blend || this.inflight || now - this.last < this.cfg.EMOTION_INTERVAL_MS) return;
    this.last = now;
    this.inflight = true;
    postJSON<{ probs: Record<string, number> }>('/api/emotion', { bs: blend.values, baseline: loadBaseline()?.values ?? null }, { quiet: true })
      .then(({ probs }) => this.update(probs))
      .catch(() => {})
      .finally(() => { this.inflight = false; });
  }

  private update(probs: Record<string, number>) {
    const a = this.cfg.EMOTION_EMA;
    const ema = Object.fromEntries(Object.entries(probs).map(([k, v]) => [k, a * v + (1 - a) * (this.ema?.[k] ?? v)]));
    this.ema = ema;
    const top = Object.keys(ema).reduce((x, y) => (ema[x] >= ema[y] ? x : y));
    if (top === this.current) { this.challenger = null; this.count = 0; }
    else if (top === this.challenger) this.count++;
    else { this.challenger = top; this.count = 1; }
    if (this.count >= 5 && ema[top] - ema[this.current] >= 0.15) { this.current = top; this.challenger = null; this.count = 0; }
    this.tally[this.current] = (this.tally[this.current] ?? 0) + 1;
    this.onTone(this.current, ema);
  }

  /** Most frequent tone since the last call, then the count starts over. */
  takeUtteranceTone() {
    const best = Object.keys(this.tally).reduce((x, y) => (this.tally[x] >= this.tally[y] ? x : y), this.current);
    this.tally = {};
    return best;
  }
}
