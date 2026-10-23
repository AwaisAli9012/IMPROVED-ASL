<script lang="ts">
  import { shortcuts } from '../stores/shortcuts.svelte';

  let dlg: HTMLDialogElement;
  $effect(() => {
    if (shortcuts.open && !dlg.open) dlg.showModal();
    else if (!shortcuts.open && dlg.open) dlg.close();
  });
</script>

<dialog bind:this={dlg} aria-label="Keyboard shortcuts" onclose={() => (shortcuts.open = false)}>
  <h2>Keyboard shortcuts</h2>
  <dl class="keys">
    {#each [...shortcuts.page, ...shortcuts.global] as [key, what] (key + what)}
      <dt><kbd>{key}</kbd></dt><dd>{what}</dd>
    {/each}
  </dl>
  <button class="btn btn--sm" type="button" onclick={() => (shortcuts.open = false)}>Close</button>
</dialog>

<style>
  .keys { display: grid; grid-template-columns: auto 1fr; gap: var(--sc-2) var(--sc-5); margin: 0; }
  dd { margin: 0; color: var(--sc-ink-soft); }
  dt { text-align: right; }
  kbd { padding: 2px 8px; border: 1px solid var(--sc-hairline-strong); border-radius: var(--sc-r-sm); background: var(--sc-canvas); font-family: var(--sc-font-mono); font-size: var(--sc-t-xs); }
</style>
