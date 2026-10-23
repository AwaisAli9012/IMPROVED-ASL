<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { Award, BarChart3, Check, Lock, Play, RotateCcw } from '@lucide/svelte';
  import { getJSON } from '$lib/api';
  import CountUp from '$lib/components/CountUp.svelte';
  import { requireProfile } from '$lib/stores/profile.svelte';

  type Unit = { id: string; title: string; kind: string; state: 'locked' | 'open' | 'passed'; items: unknown[]; mastered: number };
  const GROUPS = [
    ['Basics', (u: Unit) => u.id === 'start'],
    ['Alphabet and numbers', (u: Unit) => /^abc\d$|^num$/.test(u.id)],
    ['Core words', (u: Unit) => u.kind === 'word' && !u.id.startsWith('ext_') && u.id !== 'more'],
    ['More words', (u: Unit) => u.id.startsWith('ext_') || u.id === 'more'],
    ['Sentences', (u: Unit) => u.kind === 'sentence'],
  ] as const;
  const SEEN = 'sb.passed.seen';
  const R = 20, C = 2 * Math.PI * R;

  let data = $state<any>(null);
  let fresh = $state<string[]>([]);

  onMount(async () => {
    const me = await requireProfile();
    data = await getJSON(`/api/learn/${me.id}`);
    try { // a unit that became open since the last visit gets a one-time entrance
      const passed: string[] = data.units.filter((u: Unit) => u.state === 'passed').map((u: Unit) => u.id);
      const seen: string[] | null = JSON.parse(sessionStorage.getItem(SEEN) ?? 'null');
      if (seen) fresh = data.units.filter((u: Unit, i: number) => u.state === 'open' && data.units[i - 1] && seen.includes(data.units[i - 1].id) === false && passed.includes(data.units[i - 1].id)).map((u: Unit) => u.id);
      sessionStorage.setItem(SEEN, JSON.stringify(passed));
    } catch { /* no entrance animation */ }
    await tick();
    document.querySelector('[data-state="open"]')?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  });

  const units = $derived<Unit[]>(data?.units ?? []);
  const total = $derived(units.reduce((n, u) => n + u.items.length, 0));
  const mastered = $derived(units.reduce((n, u) => n + u.mastered, 0));
  const passed = $derived(units.filter((u) => u.state === 'passed').length);
  const current = $derived(units.find((u) => u.state === 'open')?.id);
  const grouped = $derived(GROUPS.map(([name, test]) => ({ name, units: units.filter(test) })).filter((g) => g.units.length));
  const meta = (u: Unit) => (u.items.length ? `${u.mastered} of ${u.items.length} mastered` : u.kind === 'intro' ? 'Five short pages' : 'Sign whole sentences');
</script>

<svelte:head><title>Learn · SignBridge</title></svelte:head>

<section class="stack" aria-label="Your course">
  <div class="hero-line">
    <div class="stack">
      <h1>{data ? `Hello, ${data.profile.name}` : 'Learn sign language'}</h1>
      <p class="muted">From setting up your camera to signing full sentences.</p>
    </div>
    {#if data}
      <div class="stats">
        <div class="stat"><b><CountUp value={data.profile.xp} /></b><span class="label">XP</span></div>
        <div class="stat"><b><CountUp value={data.profile.streak} /></b><span class="label">Day streak</span></div>
        <div class="stat"><b>{mastered}/{total}</b><span class="label">Signs mastered</span></div>
      </div>
    {/if}
  </div>
  <div class="bar" role="progressbar" aria-label="Course progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow={units.length ? Math.round((100 * passed) / units.length) : 0} style:--p={units.length ? passed / units.length : 0}><i></i></div>
  {#if data}
    <div class="row">
      {#if data.due}<a class="btn btn--primary" href="/learn/review"><RotateCcw />Review {data.due} due</a>{/if}
      {#if data.complete}<a class="btn btn--primary" href="/certificate"><Award />Get your certificate</a>{/if}
      <a class="btn" href="/progress"><BarChart3 />Your progress</a>
    </div>
    {#each grouped as g (g.name)}
      <details class="group" open={!current || g.units.some((u) => u.id === current)}>
        <summary><h2>{g.name}</h2><span class="muted num">{g.units.filter((u) => u.state === 'passed').length} of {g.units.length} done</span></summary>
        <ol class="path" aria-label={g.name}>
          {#each g.units as u (u.id)}
            <li class="unit" class:fresh={fresh.includes(u.id)} data-state={u.state}>
              <span class="unit__icon">
                {#if u.items.length}
                  <svg class="pring" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="24" r={R} class="pring__bg" /><circle cx="24" cy="24" r={R} class="pring__fg" stroke-dasharray={C} stroke-dashoffset={C * (1 - u.mastered / u.items.length)} /></svg>
                {/if}
                {#if u.state === 'locked'}<Lock />{:else if u.state === 'passed'}<Check />{:else}<Play />{/if}
              </span>
              <div><div class="unit__title">{u.title}</div><div class="muted">{meta(u)}</div></div>
              {#if u.state === 'locked'}
                <button class="btn btn--sm" type="button" disabled>Locked</button>
              {:else}
                <a class="btn btn--sm" class:btn--primary={u.state === 'open'} href="/learn/{u.id}">{u.state === 'open' ? (u.mastered ? 'Continue' : 'Start') : 'Practise again'}</a>
              {/if}
            </li>
          {/each}
        </ol>
      </details>
    {/each}
  {:else}
    {#each Array(6) as _, i (i)}<div class="skeleton" style="height: 4rem"></div>{/each}
  {/if}
</section>

<style>
  .group > summary { display: flex; align-items: baseline; justify-content: space-between; gap: var(--sc-4); padding: var(--sc-3) 0; cursor: pointer; list-style: none; border-bottom: 1px solid var(--sc-hairline-strong); }
  .group > summary::-webkit-details-marker { display: none; }
  .unit { position: relative; }
  /* the path: a thin line between nodes, amber once the unit is passed */
  .unit:not(:last-child)::after { content: ''; position: absolute; left: 21px; top: calc(var(--sc-4) + 44px); bottom: calc(-1 * var(--sc-4)); width: 2px; background: var(--sc-hairline-strong); }
  .unit[data-state='passed']:not(:last-child)::after { background: var(--sc-accent); }
  .unit__icon { position: relative; }
  .pring { position: absolute; inset: -2px; width: 48px; height: 48px; transform: rotate(-90deg); pointer-events: none; }
  .pring circle { fill: none; stroke-width: 3; }
  .pring__bg { stroke: transparent; }
  .pring__fg { stroke: var(--sc-accent); stroke-linecap: round; }
  .unit[data-state='open'] .pring__fg, .unit[data-state='passed'] .pring__fg { stroke: var(--sc-ink); opacity: 0.5; }
  .unit[data-state='open'] .unit__icon { animation: breathe 2.4s ease-in-out infinite; }
  @keyframes breathe { 50% { transform: scale(1.06); } }
  .fresh .unit__icon { animation: unlock 0.6s var(--sc-ease-out) both; }
  @keyframes unlock { from { transform: scale(0.95); opacity: 0; } }
</style>
