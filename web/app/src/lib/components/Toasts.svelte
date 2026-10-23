<script lang="ts">
  import { fly } from 'svelte/transition';
  import { X } from '@lucide/svelte';
  import { dismiss, toasts } from '../stores/toasts.svelte';
  import { dur, ease } from '../motion';
</script>

<div class="toasts" role="status">
  {#each toasts.list as t (t.id)}
    <div class="toast" data-kind={t.kind} transition:fly={{ y: 16, duration: dur(240), easing: ease }}>
      <span>{t.msg}</span>
      {#if t.kind === 'error'}
        <button type="button" class="toast__x" aria-label="Dismiss" onclick={() => dismiss(t.id)}><X size={16} /></button>
      {:else}
        <i class="toast__bar" aria-hidden="true"></i>
      {/if}
    </div>
  {/each}
</div>

<style>
  .toast { position: relative; display: flex; align-items: center; gap: var(--sc-3); overflow: hidden; pointer-events: auto; }
  .toast__x { display: grid; place-items: center; width: 28px; height: 28px; border: 0; border-radius: 50%; background: transparent; color: inherit; cursor: pointer; }
  .toast__bar { position: absolute; inset: auto 0 0; height: 2px; background: currentColor; opacity: 0.35; transform-origin: left; animation: drain 3.6s linear forwards; }
  .toast:hover .toast__bar { animation-play-state: paused; }
  @keyframes drain { to { transform: scaleX(0); } }
</style>
