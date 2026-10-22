// Shared helpers: fetch wrappers, toasts, safe DOM building, and the profile picker used by every page.

let cfg;
/** Server constants, fetched once. The single source of truth is signbridge/config.py. */
export const config = async () => (cfg ??= await getJSON('/api/config'));

async function request(url, init, { quiet = false, retry = true } = {}) {
  try {
    const r = await fetch(url, init);
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw Object.assign(new Error(data.error || `Request failed (${r.status})`), { status: r.status });
    return data;
  } catch (e) {
    if (retry && e.status === undefined) return request(url, init, { quiet, retry: false }); // network error: once more
    if (!quiet) toast(e.message, 'error');
    throw e;
  }
}
export const getJSON = (url, opts) => request(url, undefined, opts);
export const postJSON = (url, body, opts) =>
  request(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, opts);

/** Build DOM without innerHTML, so names and transcripts can never inject markup. */
export function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') n.className = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (typeof v === 'boolean' && /^(aria|data)-/.test(k)) n.setAttribute(k, String(v)); // aria-* and data-* need "true" and "false", not an empty string
    else if (v === true) n.setAttribute(k, '');
    else if (v !== false && v != null) n.setAttribute(k, v);
  }
  n.append(...kids.flat().filter((x) => x != null && x !== false));
  return n;
}
/** replaceChildren that skips null and false, so optional parts can be written inline. */
export const fill = (node, ...kids) => node.replaceChildren(...kids.flat().filter((k) => k != null && k !== false));
export const icon = (name) => el('i', { 'data-lucide': name });
export const refreshIcons = () => window.lucide?.createIcons();

export function announce(text) {
  const live = document.getElementById('announce');
  if (live) live.textContent = text;
}

export function toast(msg, kind = 'info') {
  const t = el('div', { class: 'toast', 'data-kind': kind }, msg);
  document.getElementById('toasts')?.append(t);
  announce(msg);
  setTimeout(() => t.remove(), 3600);
}

// ---- profile ---------------------------------------------------------------
const KEY = 'sb.profile';
export function getProfile() {
  try { return JSON.parse(localStorage.getItem(KEY)); } catch { return null; }
}
function setProfile(p) {
  try { localStorage.setItem(KEY, JSON.stringify({ id: p.id, name: p.name })); } catch { /* private mode: profile lasts for the page only */ }
  const name = document.getElementById('profile-name');
  if (name) name.textContent = p.name;
}

/** Open the profile dialog. Resolves with the chosen profile; with `required` it cannot be dismissed. */
export function pickProfile({ required = false } = {}) {
  const dlg = document.getElementById('profile-dialog');
  return new Promise(async (resolve) => {
    const profiles = await getJSON('/api/profiles').catch(() => []);
    const choose = (p) => { setProfile(p); dlg.close(); resolve(p); };
    const input = el('input', { class: 'field', name: 'name', placeholder: 'Your name', maxlength: 40, 'aria-label': 'Your name', required: true, autocomplete: 'given-name' });
    const form = el('form', { class: 'row', onsubmit: async (e) => {
      e.preventDefault();
      const name = input.value.trim();
      if (name) choose(await postJSON('/api/profiles', { name }));
    } }, input, el('button', { class: 'btn btn--primary', type: 'submit' }, 'Continue'));
    fill(dlg,
      el('h2', {}, 'Who is learning?'),
      el('p', { class: 'muted' }, 'Progress is saved on this computer under your name.'),
      profiles.length ? el('div', { class: 'row' }, profiles.map((p) => el('button', { class: 'chip chip--plain', type: 'button', onclick: () => choose(p) }, p.name))) : null,
      form,
      required ? null : el('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: () => { dlg.close(); resolve(getProfile()); } }, 'Close'));
    dlg.oncancel = required ? (e) => e.preventDefault() : null;
    dlg.showModal();
    input.focus();
  });
}

/** Every page calls this first: icons, profile chip. Returns the current profile or null. */
export function initShell({ requireProfile = false } = {}) {
  applyPrefs();
  refreshIcons();
  const p = getProfile();
  if (p) document.getElementById('profile-name').textContent = p.name;
  document.getElementById('profile-btn')?.addEventListener('click', () => pickProfile());
  return p || (requireProfile ? pickProfile({ required: true }) : null);
}

/** Camera status dot in the top bar. */
export function setCameraState(state) {
  const dot = document.getElementById('camdot');
  if (!dot) return;
  dot.dataset.state = state;
  dot.setAttribute('aria-label', { on: 'Camera on', error: 'Camera problem', off: 'Camera off' }[state] ?? 'Camera off');
}

// ---- preferences (shared by every page; Settings edits them) ------------------------------------
const PREFS_KEY = 'sb.prefs';
const DEFAULT_PREFS = { theme: 'dark', textSize: 'm', reduceMotion: false, voice: '', rate: 1, camera: '', avatarSpeed: 1, mirror: true, captionSize: 3 };
const TEXT_SIZES = { s: '93.75%', m: '100%', l: '112.5%', xl: '125%' };
export function getPrefs() {
  try { return { ...DEFAULT_PREFS, ...JSON.parse(localStorage.getItem(PREFS_KEY)) }; } catch { return { ...DEFAULT_PREFS }; }
}
export function applyPrefs() {
  const p = getPrefs(), root = document.documentElement;
  root.dataset.theme = p.theme;
  root.dataset.motion = p.reduceMotion ? 'reduced' : 'full';
  root.style.fontSize = TEXT_SIZES[p.textSize] ?? '100%';
}
export function setPref(key, value) {
  const next = { ...getPrefs(), [key]: value };
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(next)); } catch { /* private mode: lasts for this page only */ }
  applyPrefs();
  return next;
}

// ---- voice: speak and listen ----------------------------------------------------------------------
const TONE_VOICE = { neutral: [1, 1], happy: [1.08, 1.15], sad: [0.88, 0.9], angry: [1.05, 0.85], surprised: [1.1, 1.25] }; // [rate, pitch]
export function speak(text, tone = 'neutral') {
  if (!('speechSynthesis' in window)) return;
  const p = getPrefs(), [rate, pitch] = TONE_VOICE[tone] ?? TONE_VOICE.neutral;
  const u = Object.assign(new SpeechSynthesisUtterance(text), { lang: 'en-US', rate: rate * p.rate, pitch });
  const voice = speechSynthesis.getVoices().find((v) => v.voiceURI === p.voice);
  if (voice) u.voice = voice;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

/** Speech to text (Chrome and Edge). onResult({finals, interim}) fires as words arrive. Returns the recognizer, or null if unsupported. */
export function listen({ onResult, onState = () => {}, continuous = false }) {
  const Recog = window.SpeechRecognition ?? window.webkitSpeechRecognition;
  if (!Recog) return null;
  const rec = Object.assign(new Recog(), { lang: 'en-US', interimResults: true, continuous });
  rec.onstart = () => onState(true);
  rec.onend = () => onState(false);
  rec.onerror = (e) => { if (e.error !== 'no-speech' && e.error !== 'aborted') toast(`Microphone: ${e.error}`, 'error'); };
  rec.onresult = (e) => {
    const finals = [];
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) (e.results[i].isFinal ? finals.push(e.results[i][0].transcript.trim()) : (interim += e.results[i][0].transcript));
    onResult({ finals: finals.filter(Boolean), interim: interim.trim() });
  };
  rec.start();
  return rec;
}

/** Download text as a file. */
export function download(filename, text, type = 'text/plain') {
  const a = el('a', { href: URL.createObjectURL(new Blob([text], { type })), download: filename });
  a.click();
  URL.revokeObjectURL(a.href);
}
