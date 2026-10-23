<script lang="ts">
  import type { Candidate } from '../vision/recognizer';

  let { cand }: { cand: Candidate | null } = $props();
  const C = 2 * Math.PI * 18;
</script>

<div class="stage__band">
  <div class="cand">
    <div class="cand__word" aria-hidden="true">{cand?.display ?? ''}</div>
    <div class="meter" data-state={cand?.accept ? 'ok' : ''} style:--p={cand?.prob ?? 0} aria-hidden="true"><i></i></div>
  </div>
  {#if cand}
    <svg class="ring" viewBox="0 0 44 44" width="44" height="44" aria-hidden="true">
      <circle cx="22" cy="22" r="18" class="ring__bg" />
      <circle cx="22" cy="22" r="18" class="ring__fg" data-ok={cand.accept} stroke-dasharray={C} stroke-dashoffset={C * (1 - cand.prob)} />
    </svg>
  {/if}
</div>

<style>
  .ring { flex: none; transform: rotate(-90deg); }
  circle { fill: none; stroke-width: 4; }
  .ring__bg { stroke: var(--sc-hairline-strong); }
  .ring__fg { stroke: var(--warn); stroke-linecap: round; transition: stroke-dashoffset var(--sc-d-fast) var(--sc-ease-out); }
  .ring__fg[data-ok='true'] { stroke: var(--ok); }
</style>
