// Fetch wrappers, server constants, and the browser's voice. The single source of truth for constants is signbridge/config.py.
import { getPrefs } from './stores/prefs.svelte';
import { toast } from './stores/toasts.svelte';

export type Cfg = {
  POSE_IDX: number[]; LIP_IDX: number[]; LETTERS: string[]; DIGITS: string[]; EMOTIONS: string[];
  STATIC_STREAK: number; SPELL_GAP_FRAMES: number; SEG_START_FRAMES: number; SEG_END_FRAMES: number; SEG_MIN_FRAMES: number; SEG_MAX_FRAMES: number;
  BROW_RAISE_DELTA: number; BROW_DOWN_DELTA: number; EMOTION_INTERVAL_MS: number; EMOTION_EMA: number;
  thresholds: { word: number; static: number }; gemini: boolean; words: { label: string; display: string }[];
};
export type ApiError = Error & { status?: number };
type Opts = { quiet?: boolean; retry?: boolean };

async function request<T>(url: string, init: RequestInit | undefined, { quiet = false, retry = true }: Opts = {}): Promise<T> {
  try {
    const r = await fetch(url, init);
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw Object.assign(new Error(data.error || `Request failed (${r.status})`), { status: r.status });
    return data as T;
  } catch (e) {
    const err = e as ApiError;
    if (retry && err.status === undefined) return request<T>(url, init, { quiet, retry: false }); // network error: once more
    if (!quiet) toast(err.message, 'error');
    throw err;
  }
}
export const getJSON = <T = any>(url: string, opts?: Opts) => request<T>(url, undefined, opts);
export const postJSON = <T = any>(url: string, body: unknown, opts?: Opts) =>
  request<T>(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, opts);
export const deleteJSON = <T = any>(url: string, opts?: Opts) => request<T>(url, { method: 'DELETE' }, opts);

let cfg: Promise<Cfg> | undefined;
/** Server constants, fetched once. */
export const loadConfig = () => (cfg ??= getJSON<Cfg>('/api/config'));

// ---- voice: speak and listen ----------------------------------------------------------------------
const TONE_VOICE: Record<string, [number, number]> = { neutral: [1, 1], happy: [1.08, 1.15], sad: [0.88, 0.9], angry: [1.05, 0.85], surprised: [1.1, 1.25] }; // [rate, pitch]
export function speak(text: string, tone = 'neutral') {
  if (!('speechSynthesis' in window)) return;
  const p = getPrefs(), [rate, pitch] = TONE_VOICE[tone] ?? TONE_VOICE.neutral;
  const u = Object.assign(new SpeechSynthesisUtterance(text), { lang: 'en-US', rate: rate * p.rate, pitch });
  const voice = speechSynthesis.getVoices().find((v) => v.voiceURI === p.voice);
  if (voice) u.voice = voice;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

export const speechSupported = () => typeof window !== 'undefined' && ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window);

/** Speech to text (Chrome and Edge). onResult({finals, interim}) fires as words arrive. Returns the recognizer, or null if unsupported. */
export function listen({ onResult, onState = () => {}, continuous = false }: {
  onResult: (r: { finals: string[]; interim: string }) => void; onState?: (on: boolean) => void; continuous?: boolean;
}) {
  const w = window as any;
  const Recog = w.SpeechRecognition ?? w.webkitSpeechRecognition;
  if (!Recog) return null;
  const rec = Object.assign(new Recog(), { lang: 'en-US', interimResults: true, continuous });
  rec.onstart = () => onState(true);
  rec.onend = () => onState(false);
  rec.onerror = (e: { error: string }) => { if (e.error !== 'no-speech' && e.error !== 'aborted') toast(`Microphone: ${e.error}`, 'error'); };
  rec.onresult = (e: any) => {
    const finals: string[] = [];
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) (e.results[i].isFinal ? finals.push(e.results[i][0].transcript.trim()) : (interim += e.results[i][0].transcript));
    onResult({ finals: finals.filter(Boolean), interim: interim.trim() });
  };
  rec.start();
  return rec as { stop(): void };
}

/** Download text as a file. */
export function download(filename: string, text: string, type = 'text/plain') {
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(new Blob([text], { type })), download: filename });
  a.click();
  URL.revokeObjectURL(a.href);
}
