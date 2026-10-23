<script lang="ts">
  import { onMount } from 'svelte';
  import { Plus, X } from '@lucide/svelte';
  import { getJSON, postJSON, speak } from '$lib/api';
  import Avatar from '$lib/components/Avatar.svelte';

  type Group = { title: string; items: string[] };
  const MINE = 'sb.phrases';
  const read = (): string[] => { try { return JSON.parse(localStorage.getItem(MINE) ?? '[]') ?? []; } catch { return []; } };

  let groups = $state<Group[]>([]);
  let mine = $state<string[]>([]);
  let q = $state('');
  let own = $state('');
  let tab = $state('All');
  let aloud = $state(true);
  let playing = $state('');
  let figure: Avatar;
  let turn = 0;

  const all = $derived<(Group & { own?: boolean })[]>([...groups, { title: 'My phrases', items: mine, own: true }]);
  const shown = $derived(all
    .filter((g) => tab === 'All' || g.title === tab)
    .map((g) => ({ ...g, items: g.items.filter((t) => !q.trim() || t.toLowerCase().includes(q.trim().toLowerCase())) }))
    .filter((g) => g.items.length || (g.own && !q.trim())));

  onMount(async () => {
    mine = read();
    groups = await getJSON<Group[]>('/api/phrases');
    const start = new URLSearchParams(location.search).get('q'); // the command palette links here
    if (start) { q = start; say(start); }
  });

  function save(list: string[]) {
    mine = list;
    try { localStorage.setItem(MINE, JSON.stringify(list)); } catch { /* private mode: kept for this page only */ }
  }

  async function say(text: string) {
    const id = ++turn;
    playing = text;
    if (aloud) speak(text);
    const { tokens } = await postJSON('/api/gloss', { text });
    if (id !== turn) return;
    await figure.play(tokens);
    if (id === turn) playing = '';
  }

  function add(e: SubmitEvent) {
    e.preventDefault();
    const text = own.trim();
    if (text && !mine.includes(text)) { save([...mine, text]); own = ''; say(text); }
  }
</script>

<svelte:head><title>Phrases · SignBridge</title></svelte:head>

<section class="stack" aria-label="Phrasebook">
  <div class="stack"><h1>Phrases</h1><p class="muted">Tap a phrase to see it signed and hear it spoken. Useful when you need something quickly.</p></div>
  <div class="dict">
    <div class="stack">
      <input class="field" type="search" bind:value={q} placeholder="Search phrases" aria-label="Search phrases" autocomplete="off" />
      <div class="seg" role="radiogroup" aria-label="Category" style="flex-wrap: wrap">
        {#each ['All', ...all.map((g) => g.title)] as t (t)}
          <button type="button" role="radio" aria-checked={tab === t} onclick={() => (tab = t)}>{t}</button>
        {/each}
      </div>
      <div class="stack">
        {#each shown as g (g.title)}
          <div class="stack">
            <h2>{g.title}</h2>
            <div class="choice">
              {#each g.items as t (t)}
                <span class="phrase">
                  <button class="chip chip--plain" type="button" class:playing={playing === t} onclick={() => say(t)}>{t}</button>
                  {#if g.own}<button class="phrase__x" type="button" aria-label="Remove {t}" onclick={() => save(mine.filter((x) => x !== t))}><X /></button>{/if}
                </span>
              {/each}
            </div>
            {#if g.own && !g.items.length}<p class="muted">Phrases you add appear here.</p>{/if}
          </div>
        {/each}
      </div>
      <form class="composer" onsubmit={add} aria-label="Add your own phrase">
        <input class="field" bind:value={own} placeholder="Add your own phrase" maxlength="80" aria-label="Your own phrase" />
        <button class="btn" type="submit"><Plus />Add</button>
      </form>
    </div>
    <div class="stack phrase-fig">
      <Avatar bind:this={figure} />
      <label class="row" style="gap: var(--sc-2)"><input type="checkbox" bind:checked={aloud} /> <span>Speak aloud</span></label>
    </div>
  </div>
</section>

<style>
  .playing { outline: 2px solid var(--sc-accent); outline-offset: 2px; }
</style>
