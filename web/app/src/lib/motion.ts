import { prefs } from './stores/prefs.svelte';

/** Close to the scroll-craft cubic-bezier(0.23, 1, 0.32, 1). */
export const ease = (t: number) => 1 - Math.pow(1 - t, 4);

/** True when the system or the in-app setting asks for less motion. */
export const reduced = () => prefs.reduceMotion || (typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches);

/** A duration that collapses to zero under reduced motion. */
export const dur = (ms: number) => (reduced() ? 0 : ms);
