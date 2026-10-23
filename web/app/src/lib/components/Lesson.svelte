<script lang="ts">
  import { goto } from '$app/navigation';
  import { onMount, tick } from 'svelte';
  import { ArrowLeft, ArrowRight, Award } from '@lucide/svelte';
  import { getJSON, loadConfig, postJSON, type Cfg } from '../api';
  import { Recognizer } from '../vision/recognizer';
  import type { Candidate } from '../vision/recognizer';
  import type { Vision } from '../vision/vision';
  import { requireProfile } from '../stores/profile.svelte';
  import { announce, toast } from '../stores/toasts.svelte';
  import Avatar from './Avatar.svelte';
  import CameraStage from './CameraStage.svelte';
  import CandidateBand from './CandidateBand.svelte';

  let { unitId }: { unitId: string } = $props();

  type Hit = { kind: 'gloss' | 'char' | 'suggest'; label?: string; display?: string };
  type View =
    | { t: 'watch'; item: string }
    | { t: 'practise'; item: string; stuck: boolean }
    | { t: 'choose'; options: string[]; picked?: string; answer: string }
    | { t: 'signq'; item: string }
    | { t: 'intro'; i: number }
    | { t: 'sentence'; s: { english: string; gloss: string[] } }
    | { t: 'result'; score: number; passed: boolean; first: boolean; next?: string }
    | { t: 'plain'; title: string; text: string };

  const STEPS = ['Watch', 'Try it', 'Quiz'];
  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const shuffle = <T,>(a: T[]) => a.map((x) => [Math.random(), x] as const).sort((p, q) => p[0] - q[0]).map(([, x]) => x);
  const isHand = (item: string) => /^[A-Z0-9]$/.test(item);

  let cfg: Cfg;
  let me: { id: number; name: string };
  let course: any;
  let unit = $state<any>(null);
  let vision = $state<Vision>();
  let figure = $state<Avatar>();
  let recognizer: Recognizer | undefined;
  let waiting: ((e: Hit) => void) | null = null; // set while a try is open: receives recognizer events
  let pending: ((v: any) => void) | null = null;
  let dead = false;

  let view = $state<View | null>(null);
  let stepsNow = $state('Watch');
  let progress = $state(0);
  let cand = $state<Candidate | null>(null);
  let busy = $state(false);
  let fb = $state({ text: '', state: '' });
  let trying = $state(0); // increments for every open try; keys the countdown ring
  let xpPop = $state(0);
  let shake = $state(0);
  let displayOf: Record<string, string> = {};
  const show = (item: string) => displayOf[item] ?? item;

  const clicked = <T,>() => new Promise<T>((resolve) => { pending = resolve; });
  function answer(v: unknown) { const p = pending; pending = null; p?.(v); }
  function say(text: string, state = '') { fb = { text, state }; announce(text); }
  const play = (item: string, loop = true) => figure!.play([{ type: 'sign', label: item }], { loop });
  const record = (item: string, correct: boolean) => postJSON('/api/learn/attempt', { profile_id: me.id, item, correct }, { quiet: true }).catch(() => {});

  $effect(() => {
    if (!vision || recognizer) return;
    recognizer = new Recognizer(cfg, {
      onGloss: (g) => waiting?.({ kind: 'gloss', label: g.label, display: g.display }),
      onChar: (c) => waiting?.({ kind: 'char', label: c, display: c }),
      onSuggest: () => waiting?.({ kind: 'suggest' }),
      onBusy: (b) => (busy = b),
      onCandidate: (c) => (cand = c),
    });
    vision.on((ev) => recognizer!.push(ev));
  });

  /** One try on the camera: resolves {ok} on the target, {ok:false, saw} on another sign, {ok:false} on timeout. */
  function attempt(target: string, timeout = 15000) {
    recognizer!.setMode(isHand(target) ? (/[0-9]/.test(target) ? 'digits' : 'letters') : 'words');
    trying++;
    return new Promise<{ ok: boolean; saw?: string }>((resolve) => {
      const done = (r: { ok: boolean; saw?: string }) => { clearTimeout(timer); waiting = null; trying = 0; recognizer!.setMode('words'); resolve(r); };
      const timer = setTimeout(() => done({ ok: false }), timeout);
      waiting = (e) => {
        if (e.kind === 'suggest') return;
        if (e.label === target) done({ ok: true });
        else done({ ok: false, saw: e.display });
      };
    });
  }

  /** Keep trying one item until it is signed, or the learner chooses to skip after three misses. */
  async function practise(item: string) {
    let misses = 0;
    view = { t: 'practise', item, stuck: false };
    for (;;) {
      say('Waiting for your sign…');
      const r = await attempt(item);
      if (dead) return false;
      if (r.ok) { say('Yes, that is it. +10 XP', 'ok'); xpPop++; await record(item, true); await sleep(900); return true; }
      misses++; shake++;
      say(r.saw ? `That looked like “${r.saw}”. Watch it again.` : 'Time is up. Watch it again.', 'bad');
      play(item);
      if (misses >= 3) {
        view = { t: 'practise', item, stuck: true };
        if (await clicked<string>() === 'skip') { await record(item, false); return false; }
        misses = 0;
        view = { t: 'practise', item, stuck: false };
      }
    }
  }

  async function learnItem(item: string) {
    stepsNow = 'Watch';
    play(item);
    view = { t: 'watch', item };
    await clicked();
    stepsNow = 'Try it';
    await practise(item);
  }

  // ---- quiz ---------------------------------------------------------------------------------------
  function pickDistractors(item: string, pool: string[]) {
    const others = shuffle([...new Set(pool)].filter((x) => x !== item));
    const filler = shuffle(isHand(item) ? [...cfg.LETTERS, ...cfg.DIGITS] : cfg.words.map((w) => w.label)).filter((x) => x !== item && !others.includes(x));
    return [...others, ...filler].slice(0, 3);
  }

  async function chooseQuestion(item: string, pool: string[]) {
    play(item);
    const options = shuffle([item, ...pickDistractors(item, pool)]);
    fb = { text: '', state: '' };
    view = { t: 'choose', options, answer: item };
    const picked = await clicked<string>();
    view = { t: 'choose', options, answer: item, picked };
    const ok = picked === item;
    say(ok ? 'Correct' : `It was “${show(item)}”`, ok ? 'ok' : 'bad');
    await record(item, ok);
    await sleep(1100);
    return ok;
  }

  async function signQuestion(item: string) {
    figure!.stop();
    view = { t: 'signq', item };
    say('Waiting for your sign…');
    const r = await attempt(item);
    if (dead) return false;
    say(r.ok ? 'Correct' : r.saw ? `That looked like “${r.saw}”` : 'Time is up', r.ok ? 'ok' : 'bad');
    await record(item, r.ok);
    await sleep(1100);
    return r.ok;
  }

  async function quiz(items: string[]) {
    stepsNow = 'Quiz';
    const picks = shuffle(items).slice(0, 6);
    while (picks.length < 6) picks.push(items[picks.length % items.length]);
    let score = 0;
    for (const [i, item] of picks.entries()) {
      progress = (items.length + i / 6) / (items.length + 1);
      score += (i % 2 === 0 ? await chooseQuestion(item, items) : await signQuestion(item)) ? 1 : 0;
    }
    return score / 6;
  }

  async function finish(score: number) {
    figure?.stop();
    progress = 1;
    const result = await postJSON('/api/learn/unit_pass', { profile_id: me.id, unit_id: unit.id, score });
    const next = course.units[course.units.findIndex((u: any) => u.id === unit.id) + 1];
    view = { t: 'result', score, passed: result.passed, first: !!result.first_time, next: next?.id };
  }

  // ---- unit kinds ---------------------------------------------------------------------------------
  async function runIntro() {
    for (let i = 0; i < unit.pages.length;) {
      stepsNow = unit.pages[i].title;
      progress = i / unit.pages.length;
      view = { t: 'intro', i };
      i += await clicked<number>();
      if (dead) return;
    }
    await finish(1);
  }

  async function runSentences() {
    const picks = shuffle<{ english: string; gloss: string[] }>(unit.sentences).slice(0, 5);
    let right = 0;
    for (const [i, s] of picks.entries()) {
      progress = i / picks.length;
      stepsNow = 'Try it';
      view = { t: 'sentence', s };
      figure!.play(s.gloss.map((g) => ({ type: 'sign' as const, label: g })), { loop: true });
      recognizer!.setMode('words');
      const ok = await new Promise<boolean>((resolve) => {
        let k = 0;
        say(`Sign “${show(s.gloss[0])}”`);
        waiting = (e) => {
          if (e.kind === 'suggest') return;
          if (e.label !== s.gloss[k]) { k = 0; say(`That looked like “${e.display}”. Start again with “${show(s.gloss[0])}”.`, 'bad'); return; }
          k++;
          if (k === s.gloss.length) { waiting = null; resolve(true); } else say(`Good. Next: “${show(s.gloss[k])}”`, 'ok');
        };
        pending = () => { waiting = null; resolve(false); };
      });
      if (dead) return;
      right += ok ? 1 : 0;
      say(ok ? 'Sentence signed' : 'Skipped', ok ? 'ok' : '');
      await sleep(900);
    }
    await finish(right / picks.length);
  }

  async function main() {
    if (unit.kind === 'intro') return runIntro();
    if (unit.kind === 'sentence') return runSentences();
    const items: string[] = unit.items.map((i: any) => i.item);
    if (unit.kind === 'review') {
      if (!items.length) { view = { t: 'plain', title: 'Nothing due', text: 'Come back tomorrow for a review.' }; return; }
      stepsNow = 'Review';
      for (const [i, item] of items.entries()) { progress = i / items.length; play(item); await practise(item); if (dead) return; }
      figure!.stop();
      view = { t: 'plain', title: 'Review done', text: '' };
      return;
    }
    for (const [i, item] of items.entries()) { progress = i / (items.length + 1); await learnItem(item); if (dead) return; }
    await finish(await quiz(items));
  }

  onMount(() => {
    (async () => {
      me = await requireProfile();
      cfg = await loadConfig();
      displayOf = Object.fromEntries(cfg.words.map((w) => [w.label, w.display]));
      course = await getJSON(`/api/learn/${me.id}`);
      unit = course.units.find((u: any) => u.id === unitId);
      if (unitId === 'review') {
        const due = await getJSON<any[]>(`/api/learn/${me.id}/review`);
        unit = { id: 'review', title: 'Review', kind: 'review', state: 'open', items: due.map((d) => ({ item: d.item, display: d.display })) };
      }
      if (!unit || unit.state === 'locked') { toast('Finish the previous unit first.'); setTimeout(() => goto('/learn'), 1200); return; }
      if (unit.kind !== 'intro') { await tick(); while (!recognizer && !dead) await sleep(10); } // the camera stage mounts once the unit is known
      await main();
    })().catch((e) => console.error(e));
    return () => { dead = true; waiting = null; pending = null; };
  });

  const pages = $derived(unit?.pages ?? []);
  const stepNames = $derived(unit?.kind === 'intro' ? pages.map((p: any) => p.title) : unit?.kind === 'review' ? ['Review'] : STEPS);
  const R = 18, C = 2 * Math.PI * R;
</script>

<svelte:head><title>{unit?.title ?? 'Lesson'} · SignBridge</title></svelte:head>

<div class="stack">
  <div class="row row--between">
    <a class="btn btn--ghost btn--sm" href="/learn"><ArrowLeft />Course</a>
    <ol class="steps" aria-label="Lesson steps">
      {#each stepNames as n (n)}<li aria-current={n === stepsNow ? 'step' : undefined}>{n}</li>{/each}
    </ol>
  </div>
  <h1>{unit?.title ?? ''}</h1>
  <div class="bar" style:--p={progress}><i></i></div>

  {#if unit && unit.kind !== 'intro'}
  <div class="lesson">
    <section class="stack" aria-label="Watch"><Avatar bind:this={figure} /></section>
    <section aria-label="Your turn">
      <CameraStage bind:vision {busy}>
        {#snippet band()}<CandidateBand {cand} />{/snippet}
        {#if trying}
          {#key trying}
            <svg class="count" viewBox="0 0 44 44" width="44" height="44" aria-hidden="true"><circle cx="22" cy="22" r={R} class="count__bg" /><circle cx="22" cy="22" r={R} class="count__fg" stroke-dasharray={C} /></svg>
          {/key}
        {/if}
      </CameraStage>
    </section>
  </div>
  {/if}

  <div class="panel" aria-live="polite">
    {#if view?.t === 'watch'}
      <div class="big">{show(view.item)}</div>
      <p class="muted">Watch the sign. Slow it down if you need to, then try it yourself.</p>
      <div class="row"><button class="btn btn--primary" type="button" onclick={() => answer(1)}>I'm ready<ArrowRight /></button></div>
    {:else if view?.t === 'practise'}
      <div class="big">{show(view.item)}</div>
      {#if !view.stuck}<p class="muted">Sign it to the camera. You have about fifteen seconds.</p>{/if}
      {#key shake}<p class="feedback" class:shake={shake > 0 && fb.state === 'bad'} role="status" data-state={fb.state}>{fb.text}</p>{/key}
      {#if xpPop}{#key xpPop}<span class="xp" aria-hidden="true">+10 XP</span>{/key}{/if}
      {#if view.stuck}
        <div class="row"><button class="btn btn--primary" type="button" onclick={() => answer('again')}>Try again</button><button class="btn" type="button" onclick={() => answer('skip')}>Skip for now</button></div>
      {/if}
    {:else if view?.t === 'choose'}
      <h2>What does this sign mean?</h2>
      <div class="options">
        {#each view.options as o (o)}
          <button class="btn" type="button" disabled={!!view.picked} data-state={view.picked ? (o === view.answer ? 'ok' : o === view.picked ? 'bad' : undefined) : undefined} onclick={() => answer(o)}>{show(o)}</button>
        {/each}
      </div>
      <p class="feedback" role="status" data-state={fb.state}>{fb.text}</p>
    {:else if view?.t === 'signq'}
      <h2>Sign this word</h2>
      <div class="big">{show(view.item)}</div>
      <p class="feedback" role="status" data-state={fb.state}>{fb.text}</p>
    {:else if view?.t === 'intro'}
      {@const p = pages[view.i]}
      <h2>{p.title}</h2>
      <p class="lede" style="font-size: var(--sc-t-lg); max-width: 46ch">{p.body}</p>
      <div class="row"><button class="btn" type="button" disabled={view.i === 0} onclick={() => answer(-1)}>Back</button><button class="btn btn--primary" type="button" onclick={() => answer(1)}>{view.i === pages.length - 1 ? 'Finish' : 'Next'}</button></div>
    {:else if view?.t === 'sentence'}
      <h2>Sign this sentence</h2>
      <div class="big">{view.s.english}</div>
      <p class="muted">Signs, in order: {view.s.gloss.map(show).join(', ')}</p>
      <p class="feedback" role="status" data-state={fb.state}>{fb.text}</p>
      <div class="row"><button class="btn" type="button" onclick={() => answer('skip')}>Skip</button></div>
    {:else if view?.t === 'result'}
      {#if view.passed}
        <h2>Unit complete</h2>
        <p class="muted">{Math.round(view.score * 100)}%{view.first ? ' · +50 XP' : ''}</p>
        <div class="row">
          {#if view.next}<a class="btn btn--primary" href="/learn/{view.next}">Next unit<ArrowRight /></a>{:else}<a class="btn btn--primary" href="/certificate"><Award />Get your certificate</a>{/if}
          <a class="btn" href="/learn">Back to the course</a>
        </div>
      {:else}
        <h2>{Math.round(view.score * 100)}%. Nearly there</h2>
        <p class="muted">You need 80% to unlock the next unit. A little more practice will do it.</p>
        <div class="row"><button class="btn btn--primary" type="button" onclick={() => location.reload()}>Try again</button><a class="btn" href="/learn">Back to the course</a></div>
      {/if}
    {:else if view?.t === 'plain'}
      <h2>{view.title}</h2>
      {#if view.text}<p class="muted">{view.text}</p>{/if}
      <a class="btn btn--primary" href="/learn">Back to the course</a>
    {/if}
  </div>
</div>

<style>
  .count { position: absolute; top: var(--sc-3); right: var(--sc-3); transform: rotate(-90deg); }
  circle { fill: none; stroke-width: 4; }
  .count__bg { stroke: var(--sc-hairline-strong); }
  .count__fg { stroke: var(--sc-accent); stroke-linecap: round; animation: countdown 15s linear forwards; }
  @keyframes countdown { from { stroke-dashoffset: 0; } to { stroke-dashoffset: 113.1; } }
  .shake { animation: shake 0.3s ease-in-out; }
  @keyframes shake { 25% { transform: translateX(-6px); } 75% { transform: translateX(6px); } }
  .xp { display: inline-block; color: var(--ok); font-weight: 600; animation: rise-fade 0.9s var(--sc-ease-out) both; }
  @keyframes rise-fade { from { opacity: 1; transform: translateY(0); } to { opacity: 0; transform: translateY(-24px); } }
  @media (prefers-reduced-motion: reduce) { .shake, .xp, .count__fg { animation: none; } }
  :global(:root[data-motion='reduced']) .shake, :global(:root[data-motion='reduced']) .xp { animation: none; }
</style>
