<script lang="ts">
  import { goto } from '$app/navigation';
  import { getJSON, loadConfig } from '../api';
  import { setPref, prefs } from '../stores/prefs.svelte';
  import { palette } from '../stores/shortcuts.svelte';
  import { nextTheme } from '../theme';

  type Item = { id: string; label: string; hint: string; run: () => void };
  let dlg: HTMLDialogElement;
  let input: HTMLInputElement;
  let q = $state('');
  let active = $state(0);
  let signs = $state<Item[]>([]);
  let phrases = $state<Item[]>([]);

  const go = (path: string) => () => { goto(path); };
  const pages: Item[] = [
    ['Converse', '/app'], ['Learn', '/learn'], ['Phrases', '/phrasebook'], ['Dictionary', '/dictionary'], ['Metrics', '/metrics'],
    ['Progress', '/progress'], ['Live captions', '/captions'], ['Settings', '/settings'], ['About', '/about'], ['Certificate', '/certificate'],
  ].map(([label, path]) => ({ id: `page:${path}`, label, hint: 'Page', run: go(path) }));
  const actions: Item[] = [
    { id: 'act:theme', label: 'Switch theme', hint: 'Action', run: () => nextTheme() },
    { id: 'act:motion', label: 'Toggle reduce motion', hint: 'Action', run: () => setPref('reduceMotion', !prefs.reduceMotion) },
    { id: 'act:start', label: 'Start the course', hint: 'Action', run: go('/learn/start') },
  ];

  // Signs and phrases load the first time the palette opens.
  let loaded = false;
  async function load() {
    if (loaded) return;
    loaded = true;
    const cfg = await loadConfig();
    const sign = (label: string, display: string): Item => ({ id: `sign:${label}`, label: display, hint: 'Sign', run: go(`/dictionary#${label}`) });
    signs = [...cfg.words.map((w) => sign(w.label, w.display)), ...[...cfg.LETTERS, ...cfg.DIGITS].map((c) => sign(c, c))];
    const groups = await getJSON<{ title: string; items: string[] }[]>('/api/phrases', { quiet: true }).catch(() => []);
    phrases = groups.flatMap((g) => g.items.map((p) => ({ id: `phrase:${p}`, label: p, hint: 'Phrase', run: go(`/phrasebook?q=${encodeURIComponent(p)}`) })));
  }

  /** Prefix matches first, then substring matches, at most 8. */
  const results = $derived.by(() => {
    const s = q.trim().toLowerCase();
    const all = [...pages, ...actions, ...signs, ...phrases];
    if (!s) return [...pages, ...actions].slice(0, 8);
    const starts = all.filter((i) => i.label.toLowerCase().startsWith(s));
    const has = all.filter((i) => !i.label.toLowerCase().startsWith(s) && i.label.toLowerCase().includes(s));
    return [...starts, ...has].slice(0, 8);
  });

  $effect(() => {
    if (palette.open && !dlg.open) { q = ''; active = 0; dlg.showModal(); input.focus(); load(); }
    else if (!palette.open && dlg.open) dlg.close();
  });
  $effect(() => { results; active = 0; });

  function choose(i: Item | undefined) {
    if (!i) return;
    palette.open = false;
    i.run();
  }

  function key(e: KeyboardEvent) {
    if (e.key === 'ArrowDown') { e.preventDefault(); active = (active + 1) % Math.max(results.length, 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); active = (active - 1 + results.length) % Math.max(results.length, 1); }
    else if (e.key === 'Enter') { e.preventDefault(); choose(results[active]); }
  }
</script>

<dialog bind:this={dlg} class="palette" aria-label="Search" onclose={() => (palette.open = false)}>
  <input bind:this={input} bind:value={q} class="field palette__input" role="combobox" aria-expanded="true" aria-controls="palette-list"
    aria-activedescendant={results[active] ? `opt-${active}` : undefined} aria-label="Search pages, signs and phrases" placeholder="Search pages, signs and phrases" autocomplete="off" onkeydown={key} />
  <ul id="palette-list" role="listbox" class="palette__list" aria-label="Results">
    {#each results as r, i (r.id)}
      <li id="opt-{i}" role="option" aria-selected={i === active}>
        <button type="button" tabindex="-1" onclick={() => choose(r)} onmousemove={() => (active = i)} data-active={i === active}>
          <span>{r.label}</span><span class="label">{r.hint}</span>
        </button>
      </li>
    {:else}
      <li class="muted palette__none">Nothing matches.</li>
    {/each}
  </ul>
</dialog>

<style>
  .palette { width: min(34rem, 92vw); padding: var(--sc-4); margin-top: 12vh; }
  .palette__input { width: 100%; }
  .palette__list { list-style: none; margin: var(--sc-3) 0 0; padding: 0; }
  .palette__list button { display: flex; width: 100%; align-items: center; justify-content: space-between; min-height: 44px; padding: 0 var(--sc-4); border: 0; border-radius: var(--sc-r-md); background: transparent; text-align: left; cursor: pointer; }
  .palette__list button[data-active='true'] { background: var(--sc-canvas); box-shadow: inset 3px 0 0 var(--sc-accent); }
  .palette__none { padding: var(--sc-3) var(--sc-4); }
</style>
