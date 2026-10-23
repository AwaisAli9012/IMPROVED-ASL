import { reduced } from './motion';
import { prefs, setPref } from './stores/prefs.svelte';

const ORDER = ['dark', 'light', 'contrast'] as const;

/** Switch to the next theme, cross-fading through a view transition when the browser has one. */
export function nextTheme() {
  const next = ORDER[(ORDER.indexOf(prefs.theme) + 1) % ORDER.length];
  const apply = () => setPref('theme', next);
  if (document.startViewTransition && !reduced()) document.startViewTransition(apply);
  else apply();
}
