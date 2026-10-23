<script lang="ts">
  import { onMount } from 'svelte';
  import { Avatar, type Token } from '../avatar/avatar';
  import { prefs } from '../stores/prefs.svelte';

  let { controls = true, label = 'Signing figure', ratio = '16 / 10', onspeed }: {
    controls?: boolean; label?: string; ratio?: string; onspeed?: () => void;
  } = $props();

  let canvas: HTMLCanvasElement;
  let avatar: Avatar | undefined;
  let speed = $state(prefs.avatarSpeed);
  let mirror = $state(prefs.mirror);
  let caption = $state('');

  onMount(() => {
    avatar = new Avatar(canvas);
    return () => avatar?.destroy();
  });

  $effect(() => { avatar?.setMirror(mirror); });

  export const play = (tokens: Token[], opts: { loop?: boolean; onWord?: (w: string, i: number) => void } = {}) =>
    avatar!.play(tokens, { ...opts, onWord: (w, i) => { caption = w; opts.onWord?.(w, i); } }).then((done) => { if (!opts.loop) caption = ''; return done; });
  export const stop = () => { avatar?.stop(); caption = ''; };
  export const show = (label: string) => avatar?.show(label);

  function setSpeed(s: number) {
    speed = s;
    avatar?.setSpeed(s);
    onspeed?.();
  }
</script>

<div class="avatar" style:aspect-ratio={ratio}><!-- svelte-ignore a11y_no_interactive_element_to_noninteractive_role --><canvas bind:this={canvas} role="img" aria-label={label}></canvas></div>
<p class="caption" aria-hidden="true">{caption}</p>
{#if controls}
  <div class="row row--between">
    <div class="seg" role="radiogroup" aria-label="Signing speed">
      {#each [0.5, 1] as s (s)}
        <button type="button" role="radio" aria-checked={speed === s} onclick={() => setSpeed(s)}>{s}×</button>
      {/each}
    </div>
    <label class="row" style="gap: var(--sc-2)"><input type="checkbox" bind:checked={mirror} /> <span>Mirror me</span></label>
  </div>
{/if}
