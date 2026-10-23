<script lang="ts">
  import { onMount } from 'svelte';
  import { Award, Flame, GraduationCap, Hand, Hash, Medal, Sparkles, Star, Target, Type } from '@lucide/svelte';
  import { reveal } from '$lib/actions';
  import { getJSON } from '$lib/api';
  import CountUp from '$lib/components/CountUp.svelte';
  import { requireProfile } from '$lib/stores/profile.svelte';

  const ICONS: Record<string, typeof Award> = { first_sign: Sparkles, ten_signs: Hand, fifty_correct: Star, ten_mastered: Medal, alphabet: Type, numbers: Hash, streak3: Flame, streak7: Flame, perfect: Target, course: GraduationCap };
  const pct = new Intl.NumberFormat('en', { style: 'percent', maximumFractionDigits: 0 });
  const SEEN = 'sb.badges.seen';

  let r = $state<any>(null);
  let fresh = $state<string[]>([]); // badges earned since the last visit get a one-time entrance

  onMount(async () => {
    const me = await requireProfile();
    r = await getJSON(`/api/progress/${me.id}`);
    try {
      const seen: string[] = JSON.parse(localStorage.getItem(SEEN) ?? '[]');
      const earned: string[] = r.badges.filter((b: any) => b.earned).map((b: any) => b.id);
      fresh = earned.filter((id) => !seen.includes(id));
      localStorage.setItem(SEEN, JSON.stringify(earned));
    } catch { /* private mode: no entrance animation */ }
  });

  const peak = $derived(r ? Math.max(1, ...r.activity.map((d: any) => d.n)) : 1);
  const total = $derived(r ? r.activity.reduce((n: number, d: any) => n + d.n, 0) : 0);
</script>

<svelte:head><title>Your progress · SignBridge</title></svelte:head>

<section class="stack" aria-label="Your progress">
  {#if !r}
    <h1>Your progress</h1>
    <div class="skeleton" style="height: 14rem"></div>
  {:else}
    {@const t = r.totals}
    <div class="hero-line">
      <div class="stack"><h1>Your progress</h1><p class="muted">{r.profile.name}: {t.seen} of {t.signs} signs tried, {t.mastered} mastered.</p></div>
      <div class="stats">
        <div class="stat"><b><CountUp value={r.profile.xp} /></b><span class="label">XP</span></div>
        <div class="stat"><b>{r.profile.longest_streak} d</b><span class="label">Best streak</span></div>
        <div class="stat"><b>{t.accuracy == null ? 'n/a' : pct.format(t.accuracy)}</b><span class="label">Accuracy</span></div>
      </div>
    </div>
    <div class="grid-2">
      <div class="panel"><h2>Last 14 days</h2>
        <div class="activity bars" use:reveal role="img" aria-label="Signs practised each day for the last 14 days">
          {#each r.activity as d (d.day)}
            <div class="activity__col" title="{d.day}: {d.n} signs"><i style="--h:{d.n / peak}"></i><span class="label">{d.day.slice(8)}</span></div>
          {/each}
        </div>
        <p class="muted">{total ? `${total} signs practised in two weeks.` : 'Practise a sign in Learn and it shows up here.'}</p></div>
      <div class="panel"><h2>Weak spots</h2>
        <div class="stack">
          {#each r.weak as w (w.item)}
            <a class="row row--between" href="/dictionary#{encodeURIComponent(w.item)}"><b>{w.display}</b><span class="muted num">{w.correct} of {w.attempts} right</span></a>
          {:else}
            <p class="muted">Nothing yet. Signs you miss twice or more appear here so you can revisit them.</p>
          {/each}
        </div></div>
    </div>
    <div class="panel"><h2>Badges</h2>
      <ul class="badges">
        {#each r.badges as b (b.id)}
          {@const Icon = ICONS[b.id] ?? Award}
          <li class="badge" class:fresh={fresh.includes(b.id)} data-earned={b.earned} title={b.detail}><Icon /><b>{b.title}</b><span class="muted">{b.earned ? 'Earned' : b.detail}</span></li>
        {/each}
      </ul></div>
    <div class="panel heat">
      <h2>Every sign</h2>
      <p class="muted">Darker squares are better known. Hover or focus one for details, click to see the sign.</p>
      <div class="legend row" aria-hidden="true"><span class="label">New</span>{#each [0, 1, 2, 3, 4, 5] as n (n)}<i class="cell" style="--b:{n}"></i>{/each}<span class="label">Mastered</span></div>
      <div class="stack">
        {#each r.units as u (u.id)}
          <div class="stack"><h3>{u.title}</h3>
            <div class="cells">
              {#each u.items as i (i.item)}
                <a class="cell cell--link tip" href="/dictionary#{encodeURIComponent(i.item)}" style="--b:{i.box}" data-tip="{i.display}: {i.attempts ? `${i.correct} of ${i.attempts} right` : 'not tried yet'}, level {i.box} of 5" aria-label="{i.display}, level {i.box} of 5">{i.display.slice(0, 3)}</a>
              {/each}
            </div></div>
        {/each}
      </div>
    </div>
    <div class="row"><a class="btn btn--primary" href="/learn"><GraduationCap />Back to the course</a></div>
  {/if}
</section>

<style>
  .bars i { transform: scaleY(0); }
  .bars:global(.seen) i { transform: scaleY(var(--h)); }
  .fresh { animation: badge-in 0.5s var(--sc-ease-out) both; }
  @keyframes badge-in { from { opacity: 0; transform: scale(0.9); } }
  .heat { overflow-x: clip; } /* a tooltip near the edge is clipped instead of widening the page on a phone */
  .tip { position: relative; }
  .tip::after { content: attr(data-tip); position: absolute; z-index: 5; left: 50%; bottom: calc(100% + 6px); translate: -50% 0; width: max-content; max-width: 16rem; padding: var(--sc-1) var(--sc-3);
    border-radius: var(--sc-r-sm); background: var(--sc-ink); color: var(--sc-canvas); font-size: var(--sc-t-xs); line-height: 1.4; pointer-events: none; opacity: 0; transition: opacity var(--sc-d-fast) var(--sc-ease-out); }
  .tip:hover::after, .tip:focus-visible::after { opacity: 1; }
</style>
