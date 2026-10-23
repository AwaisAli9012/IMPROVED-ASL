<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { getJSON, loadConfig } from '$lib/api';
  import Avatar from '$lib/components/Avatar.svelte';
  import { toast } from '$lib/stores/toasts.svelte';
  import { shortcuts } from '$lib/stores/shortcuts.svelte';

  type Entry = { label: string; display: string; unit: string; topic: string };
  let entries = $state<Entry[]>([]);
  let loading = $state(true);
  let q = $state('');
  let topic = $state('All');
  let selected = $state<Entry | null>(null);
  let figure: Avatar;
  let list: HTMLElement;

  const topics = $derived(['All', ...new Set(entries.map((e) => e.topic))]);
  const shown = $derived(entries.filter((e) => (topic === 'All' || e.topic === topic) && (!q.trim() || e.display.toLowerCase().includes(q.trim().toLowerCase()) || e.label.toLowerCase().includes(q.trim().toLowerCase()))));
  const letters = $derived([...new Set(shown.map((e) => e.display[0].toUpperCase()))].filter((l) => /[A-Z]/.test(l)));

  onMount(() => {
    load();
    shortcuts.page = [['↑ ↓', 'Move through the signs'], ['/', 'Search']];
  });

  async function load() {
    const [units, cfg] = await Promise.all([getJSON<any[]>('/api/curriculum'), loadConfig()]);
    const displayOf = Object.fromEntries(cfg.words.map((w) => [w.label, w.display]));
    entries = units.flatMap((u) => (u.kind === 'hand' || u.kind === 'word' ? u.items.map((item: string) => ({
      label: item, display: displayOf[item] ?? item, unit: u.id, topic: u.kind === 'hand' ? (/[0-9]/.test(item) ? 'Numbers' : 'Alphabet') : u.title })) : []));
    loading = false;
    const start = entries.find((e) => e.label === decodeURIComponent(location.hash.slice(1)));
    if (start) select(start);
  }
  onDestroy(() => { shortcuts.page = []; });

  function select(e: Entry) {
    selected = e;
    history.replaceState(null, '', `#${encodeURIComponent(e.label)}`);
    figure.play([{ type: 'sign', label: e.label }], { loop: true });
    list?.querySelector('[aria-current="true"]')?.scrollIntoView({ block: 'nearest' });
  }

  function move(e: KeyboardEvent) {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    const i = selected ? shown.findIndex((s) => s.label === selected!.label) : -1;
    const next = shown[Math.min(shown.length - 1, Math.max(0, i + (e.key === 'ArrowDown' ? 1 : -1)))];
    if (next) select(next);
  }

  function jump(letter: string) {
    list.querySelector<HTMLElement>(`[data-initial="${letter}"]`)?.scrollIntoView({ block: 'start' });
  }

  function addPhrase(e: Entry) {
    try {
      const mine: string[] = JSON.parse(localStorage.getItem('sb.phrases') ?? '[]');
      if (!mine.includes(e.display)) localStorage.setItem('sb.phrases', JSON.stringify([...mine, e.display]));
      toast(`Added "${e.display}" to your phrases.`);
    } catch { toast('Could not save the phrase in this browser.', 'error'); }
  }

  /** Split a name around the search text so the match can be marked. */
  function parts(name: string) {
    const s = q.trim().toLowerCase(), i = s ? name.toLowerCase().indexOf(s) : -1;
    return i < 0 ? [name, '', ''] : [name.slice(0, i), name.slice(i, i + s.length), name.slice(i + s.length)];
  }
</script>

{#snippet hl(name: string)}{@const [a, m, b] = parts(name)}{a}{#if m}<mark>{m}</mark>{/if}{b}{/snippet}

<svelte:head><title>Dictionary · SignBridge</title></svelte:head>

<section class="stack" aria-label="Sign dictionary">
  <div class="stack">
    <h1>Dictionary</h1>
    <p class="muted">{loading ? 'Loading the signs…' : `${shown.length} of ${entries.length} signs`}</p>
  </div>
  <div class="row">
    <input class="field" type="search" bind:value={q} placeholder="Search a sign" aria-label="Search a sign" autocomplete="off" style="min-width: min(100%, 18rem)" />
    <div class="row" role="group" aria-label="Filter by topic">
      {#each topics as t (t)}
        <button class="chip chip--plain" type="button" aria-pressed={topic === t} onclick={() => (topic = t)}>{t}</button>
      {/each}
    </div>
  </div>
  <div class="row" role="group" aria-label="Jump to a letter">
    {#each letters as l (l)}
      <button class="btn btn--ghost btn--sm btn--icon" type="button" onclick={() => jump(l)}>{l}</button>
    {/each}
  </div>
  <div class="dict">
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <ul class="dict__list panel" bind:this={list} aria-label="Signs" style="padding: 0" onkeydown={move}>
      {#if loading}
        {#each Array(8) as _, i (i)}<li class="skeleton" style="height: 48px; margin: 6px"></li>{/each}
      {/if}
      {#each shown as e, i (e.label)}
        <li data-initial={i === 0 || shown[i - 1].display[0].toUpperCase() !== e.display[0].toUpperCase() ? e.display[0].toUpperCase() : undefined}>
          <button class="dict__item" type="button" aria-current={e.label === selected?.label} onclick={() => select(e)}>
            <span>{@render hl(e.display)}</span><span class="label">{e.topic}</span>
          </button>
        </li>
      {:else}
        {#if !loading}<li class="muted" style="padding: var(--sc-4)">No sign matches.</li>{/if}
      {/each}
    </ul>
    <div class="stack">
      <Avatar bind:this={figure} onspeed={() => selected && select(selected)} />
      <div class="row row--between">
        <div class="big">{selected?.display ?? ''}</div>
        {#if selected}
          <div class="row">
            <button class="btn btn--ghost" type="button" onclick={() => addPhrase(selected!)}>Add to my phrases</button>
            <a class="btn" href="/learn/{selected.unit}">Practise this</a>
          </div>
        {/if}
      </div>
    </div>
  </div>
</section>

<style>
  mark { background: color-mix(in oklab, var(--sc-accent) 35%, transparent); color: inherit; border-radius: 3px; }
</style>
