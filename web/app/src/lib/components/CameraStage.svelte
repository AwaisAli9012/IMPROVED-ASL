<script lang="ts">
  import { onMount, type Snippet } from 'svelte';
  import { Camera } from '@lucide/svelte';
  import { Vision } from '../vision/vision';

  let { vision = $bindable(), busy = false, onready, top, band, children }: {
    vision?: Vision; busy?: boolean; onready?: () => void | Promise<void>; top?: Snippet; band?: Snippet; children?: Snippet;
  } = $props();

  let video: HTMLVideoElement;
  let overlay: HTMLCanvasElement;
  let shown = $state(true);
  let title = $state('Sign to start talking');
  let text = $state('SignBridge reads hand, body and face points from your camera. The video stays on this computer.');
  let label = $state('Start camera');
  let loading = $state(false);

  function say(t: string, x: string, b: string) { title = t; text = x; label = b; shown = true; }

  export async function start() {
    loading = true;
    say('Starting the camera…', 'Loading the hand, body and face models. This takes a few seconds.', 'Starting…');
    try {
      await vision!.start();
      shown = false;
      await onready?.();
    } catch (e) {
      const err = e as Error;
      const denied = err.name === 'NotAllowedError' || err.name === 'SecurityError';
      say(denied ? 'Camera access is blocked' : 'The camera did not start',
        denied ? 'Click the camera icon in the address bar, choose Allow, then try again.' : err.name === 'NotFoundError' ? 'No camera was found on this computer.' : err.message, 'Try again');
    } finally { loading = false; }
  }

  onMount(() => {
    const v = new Vision(video, overlay);
    vision = v;
    navigator.permissions?.query({ name: 'camera' as PermissionName }).then((p) => { if (p.state === 'granted') start(); }).catch(() => {});
    return () => v.stop(); // leave no camera light on after navigating away
  });
</script>

<div class="stage" aria-busy={busy}>
  <video bind:this={video} class="mirror" playsinline muted></video>
  <canvas bind:this={overlay} class="mirror" aria-hidden="true"></canvas>
  {@render top?.()}
  {@render children?.()}
  {@render band?.()}
  <div class="stage__msg" hidden={!shown}>
    <h2>{title}</h2>
    <p class="muted">{text}</p>
    <div class="row center"><button class="btn btn--primary" type="button" disabled={loading} onclick={start}><Camera />{label}</button></div>
  </div>
</div>
