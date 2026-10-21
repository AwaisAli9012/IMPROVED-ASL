// Turns the stream of vision frames into signs, spelled words, tone and question markers.
// Every threshold comes from /api/config (signbridge/config.py); nothing is hard-coded here.
import { postJSON } from './api.js';

const BASELINE_KEY = 'sb.baseline';
export const loadBaseline = () => { try { return JSON.parse(localStorage.getItem(BASELINE_KEY)); } catch { return null; } };
const saveBaseline = (b) => { try { localStorage.setItem(BASELINE_KEY, JSON.stringify(b)); } catch { /* private mode: baseline lasts for the page only */ } };

/** Average the user's neutral face over `ms` and keep it as the baseline. */
export function calibrateNeutral(vision, ms = 2000) {
  return new Promise((resolve) => {
    const sum = [];
    let n = 0, names = null;
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

const mean = (b, keys) => keys.reduce((s, k) => s + b.values[b.names.indexOf(k)], 0) / keys.length;
const RAISE = ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'];
const DOWN = ['browDownLeft', 'browDownRight'];

export class Recognizer {
  /** handlers: onGloss, onSuggest(top3, question), onCandidate({display, prob, accept} | null), onBusy(bool), onChar(c, chars), onError(e) */
  constructor(cfg, handlers) {
    this.cfg = cfg;
    this.h = handlers;
    this.baseline = loadBaseline();
    this.pending = 0;
    this.chain = Promise.resolve();
    this.setMode('words');
  }

  setMode(mode) {
    this.mode = mode;
    Object.assign(this, { state: 'idle', recent: [], seg: [], gap: 0, run: 0, chars: '', noHand: 0, streak: 0, top: null, committed: null, inflight: false });
    this.brow = { raise: 0, down: 0, n: 0 };
    this.h.onCandidate?.(null);
  }

  push(ev) { return this.mode === 'words' ? this.#words(ev) : this.#static(ev); }

  // ---- words: a sign starts when a hand appears and ends when hands rest ------------------------
  #words({ frame, blend }) {
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
    this.#brows(blend);
    this.gap = hasHand ? 0 : this.gap + 1;
    if (this.gap >= c.SEG_END_FRAMES || this.seg.length >= c.SEG_MAX_FRAMES) this.endSegment();
  }

  #brows(blend) {
    if (!blend || !this.baseline) return; // without a calibrated neutral face there are no grammar markers
    const raise = mean(blend, RAISE) - mean(this.baseline, RAISE);
    const down = mean(blend, DOWN) - mean(this.baseline, DOWN);
    this.brow.n++;
    if (raise > this.cfg.BROW_RAISE_DELTA) this.brow.raise++;
    if (down > this.cfg.BROW_DOWN_DELTA) this.brow.down++;
  }

  #question() {
    const { raise, down, n } = this.brow;
    return !n ? 'none' : down / n >= 0.3 ? 'wh' : raise / n >= 0.3 ? 'yesno' : 'none';
  }

  /** Finish the current sign now (also bound to the Space key). */
  endSegment() {
    if (this.state !== 'capturing') return;
    const seg = this.seg;
    const question = this.#question();
    Object.assign(this, { state: 'idle', seg: [], recent: [], run: 0 });
    this.h.onCandidate?.(null);
    if (seg.filter((f) => f.lh || f.rh).length < this.cfg.SEG_MIN_FRAMES) return; // too short to be a sign
    this.h.onBusy?.(++this.pending > 0);
    this.chain = this.chain.then(async () => { // one request at a time keeps results in signing order
      try {
        const r = await postJSON('/api/predict/word', { frames: seg }, { quiet: true });
        if (r.accept) this.h.onGloss({ type: 'sign', label: r.top[0].label, display: r.top[0].display, question });
        else this.h.onSuggest?.(r.top.slice(0, 3), question);
      } catch (e) { this.h.onError?.(e); } finally { this.h.onBusy?.(--this.pending > 0); }
    });
  }

  // ---- ABC and 123: latch a character after STATIC_STREAK confident frames in a row ----------------
  #static({ hand, aspect }) {
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
    postJSON('/api/predict/static', { hand, aspect, mode: this.mode }, { quiet: true }).then((r) => {
      const t = r.top[0];
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
  constructor(cfg, onTone) {
    this.cfg = cfg;
    this.onTone = onTone;
    this.ema = null;
    this.current = 'neutral';
    this.challenger = null;
    this.count = 0;
    this.last = 0;
    this.inflight = false;
    this.tally = {};
  }

  push(blend, now = performance.now()) {
    if (!blend || this.inflight || now - this.last < this.cfg.EMOTION_INTERVAL_MS) return;
    this.last = now;
    this.inflight = true;
    postJSON('/api/emotion', { bs: blend.values, baseline: loadBaseline()?.values ?? null }, { quiet: true })
      .then(({ probs }) => this.#update(probs))
      .catch(() => {})
      .finally(() => { this.inflight = false; });
  }

  #update(probs) {
    const a = this.cfg.EMOTION_EMA;
    this.ema = Object.fromEntries(Object.entries(probs).map(([k, v]) => [k, a * v + (1 - a) * (this.ema?.[k] ?? v)]));
    const top = Object.keys(this.ema).reduce((x, y) => (this.ema[x] >= this.ema[y] ? x : y));
    if (top === this.current) { this.challenger = null; this.count = 0; }
    else if (top === this.challenger) this.count++;
    else { this.challenger = top; this.count = 1; }
    if (this.count >= 5 && this.ema[top] - this.ema[this.current] >= 0.15) { this.current = top; this.challenger = null; this.count = 0; }
    this.tally[this.current] = (this.tally[this.current] ?? 0) + 1;
    this.onTone(this.current, this.ema);
  }

  /** Most frequent tone since the last call, then the count starts over. */
  takeUtteranceTone() {
    const best = Object.keys(this.tally).reduce((x, y) => (this.tally[x] >= this.tally[y] ? x : y), this.current);
    this.tally = {};
    return best;
  }
}
