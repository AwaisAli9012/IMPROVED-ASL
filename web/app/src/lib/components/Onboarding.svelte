<script lang="ts">
  import { onMount } from 'svelte';
  import { Circle, CircleCheck } from '@lucide/svelte';
  import { calibrateNeutral } from '../vision/recognizer';
  import type { Vision } from '../vision/vision';

  let { vision, neutralOnly = false, onfinish }: { vision: Vision; neutralOnly?: boolean; onfinish: () => void } = $props();

  const DONE = 'sb.onboarded';
  let dlg: HTMLDialogElement;
  // svelte-ignore state_referenced_locally
  let step = $state<'position' | 'neutral' | 'done'>(neutralOnly ? 'neutral' : 'position');
  let shoulders = $state(false);
  let hand = $state(false);
  let counting = $state(false);
  let found = $state(true);

  onMount(() => {
    dlg.show(); // non-modal: the card sits over the camera view instead of dimming it
    const stop = vision.on(({ frame }) => {
      const p = frame.pose;
      shoulders = !!p && [p[1], p[2]].every(([x, y]) => x > 0 && x < 1 && y > 0 && y < 1);
      if (frame.lh || frame.rh) hand = true;
    });
    return stop;
  });

  function finish() {
    try { localStorage.setItem(DONE, '1'); } catch { /* shown again next visit */ }
    dlg.close();
    onfinish();
  }

  async function capture() {
    counting = true;
    found = !!(await calibrateNeutral(vision, 2000));
    counting = false;
    step = 'done';
  }

  const dots = $derived(neutralOnly ? [] : ['position', 'neutral', 'done']);
</script>

<dialog bind:this={dlg} id="onboard" aria-label="Set up SignBridge">
  {#if dots.length}
    <div class="dots" aria-hidden="true">{#each dots as d (d)}<i data-on={d === step}></i>{/each}</div>
  {/if}
  {#if step === 'position'}
    <h2>Check your position</h2>
    <p class="muted">Sit so your shoulders and hands are visible, with light on your face.</p>
    <ul class="stack" style="list-style: none; margin: 0; padding: 0">
      <li class="row">{#if shoulders}<CircleCheck />{:else}<Circle />{/if}Both shoulders in view</li>
      <li class="row">{#if hand}<CircleCheck />{:else}<Circle />{/if}Raise a hand</li>
    </ul>
    <div class="row">
      <button class="btn btn--primary" type="button" disabled={!(shoulders && hand)} onclick={() => (step = 'neutral')}>Next</button>
      <button class="btn btn--ghost btn--sm" type="button" onclick={() => (step = 'neutral')}>Skip</button>
    </div>
  {:else if step === 'neutral'}
    <h2>Relax your face</h2>
    <p class="muted">Look at the screen with a neutral expression. SignBridge compares later expressions with this one to read tone and raised eyebrows.</p>
    <div class="row">
      <button class="btn btn--primary" type="button" disabled={counting} onclick={capture}>
        {#if counting}<svg class="ring" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><circle cx="12" cy="12" r="9" /></svg>Hold still…{:else}Capture neutral face{/if}
      </button>
      <button class="btn btn--ghost btn--sm" type="button" onclick={finish}>{neutralOnly ? 'Close' : 'Skip'}</button>
    </div>
  {:else}
    <h2>{found ? 'All set' : 'No face found'}</h2>
    <p class="muted">{found ? 'Your tone and question markers are now tuned to your face.' : 'Face the camera and try Recalibrate on the stage.'}</p>
    <button class="btn btn--primary" type="button" onclick={finish}>Start signing</button>
  {/if}
</dialog>

<style>
  .dots { display: flex; gap: var(--sc-2); }
  .dots i { width: 8px; height: 8px; border-radius: 50%; background: var(--sc-hairline-strong); transition: background-color var(--sc-d-base) var(--sc-ease-out); }
  .dots i[data-on='true'] { background: var(--sc-accent); }
  .ring { transform: rotate(-90deg); }
  .ring circle { fill: none; stroke: currentColor; stroke-width: 3; stroke-dasharray: 56.5; animation: fill 2s linear forwards; }
  @keyframes fill { from { stroke-dashoffset: 56.5; } to { stroke-dashoffset: 0; } }
</style>
