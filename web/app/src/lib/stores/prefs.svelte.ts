// Preferences shared by every page. Same localStorage key as the old app, so existing settings carry over.
const KEY = 'sb.prefs';
export type Prefs = {
  theme: 'dark' | 'light' | 'contrast'; textSize: 's' | 'm' | 'l' | 'xl'; reduceMotion: boolean; voice: string; rate: number;
  camera: string; avatarSpeed: number; mirror: boolean; captionSize: number;
};
const DEFAULTS: Prefs = { theme: 'dark', textSize: 'm', reduceMotion: false, voice: '', rate: 1, camera: '', avatarSpeed: 1, mirror: true, captionSize: 3 };
const TEXT_SIZES = { s: '93.75%', m: '100%', l: '112.5%', xl: '125%' };

function read(): Prefs {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) ?? 'null') }; } catch { return { ...DEFAULTS }; }
}

export const prefs = $state<Prefs>(read());

export function applyPrefs() {
  const root = document.documentElement;
  root.dataset.theme = prefs.theme;
  root.dataset.motion = prefs.reduceMotion ? 'reduced' : 'full';
  root.style.fontSize = TEXT_SIZES[prefs.textSize] ?? '100%';
}

export function setPref<K extends keyof Prefs>(key: K, value: Prefs[K]) {
  prefs[key] = value;
  try { localStorage.setItem(KEY, JSON.stringify({ ...prefs })); } catch { /* private mode: lasts for this page only */ }
  applyPrefs();
}

export const getPrefs = (): Prefs => ({ ...prefs });
