<script lang="ts">
  import { flip } from 'svelte/animate';
  import { fly } from 'svelte/transition';
  import { onMount, tick } from 'svelte';
  import { Angry, Download, Frown, Gauge, Keyboard, Meh, Mic, Play, ScanFace, Send, Smile, Sparkles, Trash2, Undo2, Volume2, X } from '@lucide/svelte';
  import { download, listen, loadConfig, postJSON, speak, speechSupported, type Cfg } from '$lib/api';
  import Avatar from '$lib/components/Avatar.svelte';
  import CameraStage from '$lib/components/CameraStage.svelte';
  import CandidateBand from '$lib/components/CandidateBand.svelte';
  import Onboarding from '$lib/components/Onboarding.svelte';
  import { dur, ease } from '$lib/motion';
  import { camera } from '$lib/stores/camera.svelte';
  import { shortcuts } from '$lib/stores/shortcuts.svelte';
  import { announce, toast } from '$lib/stores/toasts.svelte';
  import { Recognizer, ToneTracker, type Candidate, type Gloss, type Guess, type Question, type Result } from '$lib/vision/recognizer';
  import type { Vision } from '$lib/vision/vision';

  type Msg = { who: 'you' | 'them'; english: string; detail: string; tone: string; tokens?: any[]; parts?: string[] };
  const WH = new Set(['what', 'where', 'who', 'why', 'how', 'when', 'which']);
  const TONE = { neutral: [Meh, 'Neutral'], happy: [Smile, 'Happy'], sad: [Frown, 'Sad'], angry: [Angry, 'Angry'], surprised: [Sparkles, 'Surprised'] } as const;
  const QUESTION: Record<string, string> = { yesno: 'Question (yes/no)', wh: 'Question (what, where, who)' };
  type ToneName = keyof typeof TONE;

  let cfg = $state<Cfg>();
  let vision = $state<Vision>();
  let stage: CameraStage;
  let figure: Avatar;
  let recognizer: Recognizer | undefined;
  let tracker: ToneTracker | undefined;

  let mode = $state<'words' | 'letters' | 'digits'>('words');
  let glosses = $state<(Gloss & { id: number })[]>([]);
  let transcript = $state<Msg[]>([]);
  let log: HTMLElement;
  let suggest = $state<{ top: Guess[]; question: Question } | null>(null);
  let suggestTimer: ReturnType<typeof setTimeout>;
  let cand = $state<Candidate | null>(null);
  let busy = $state(false);
  let got = $state('');
  let gotTimer: ReturnType<typeof setTimeout>;
  let toneName = $state<ToneName>('neutral');
  let toneConf = $state(0);
  let last = $state<Result | null>(null);
  let inspector = $state(false);
  let fps = $state(0);
  let preview = $state('');
  let reply = $state('');
  let listening = $state(false);
  let rec: { stop(): void } | null = null;
  let calibrating = $state<'first' | 'again' | null>(null);
  let spoken = $state<{ msg: number; part: number } | null>(null);
  let nextId = 0;
  const supported = speechSupported();

  const question = $derived<Question>(glosses.some((g) => g.type === 'sign' && WH.has(g.label)) ? 'wh' : (glosses.find((g) => g.question !== 'none')?.question ?? 'none'));
  const ToneIcon = $derived(TONE[toneName][0]);

  // One state value tells the user what the app is doing at every moment.
  const status = $derived(
    camera.state !== 'on' ? 'Camera off' : busy ? 'Reading…' : got ? `Got it: ${got}` : cand ? 'Watching…' : 'Show your hands');

  onMount(() => {
    shortcuts.page = [['W A N', 'Words, ABC, 123 modes'], ['Space', 'End the sign you are making now'], ['1 2 3', 'Pick a suggestion'], ['Enter', 'Speak the sentence'],
      ['Backspace', 'Remove the last sign'], ['Esc', 'Clear, or dismiss suggestions'], ['I', 'Show or hide the model inspector']];
    const tick400 = setInterval(() => { if (vision) fps = vision.fps; }, 400);
    loadConfig().then((c) => (cfg = c));
    return () => { shortcuts.page = []; clearInterval(tick400); clearTimeout(suggestTimer); clearTimeout(gotTimer); rec?.stop(); };
  });

  // Wire the recognizer once both the constants and the camera stage exist.
  $effect(() => {
    if (!cfg || !vision || recognizer) return;
    recognizer = new Recognizer(cfg, {
      onGloss: addGloss, onSuggest: showSuggestions, onChar: (c) => announce(c), onBusy: (b) => (busy = b), onResult: (r) => (last = r), onCandidate: (c) => (cand = c),
      onError: (e) => { if (e.status === 503) toast('The sign model is not trained yet.', 'error'); },
    });
    tracker = new ToneTracker(cfg, (name, ema) => { toneName = (name in TONE ? name : 'neutral') as ToneName; toneConf = ema[name] ?? 0; });
    vision.on((ev) => { recognizer!.push(ev); tracker!.push(ev.blend); });
    (window as any).sb = { vision, recognizer }; // handy in the console, and used by the browser checks: sb.vision.fps
  });

  // ---- building the sentence ----------------------------------------------------------------------
  function addGloss(g: Gloss) {
    glosses.push({ ...g, id: nextId++ });
    clearSuggestions();
    clearTimeout(gotTimer);
    got = g.display;
    gotTimer = setTimeout(() => (got = ''), 2000);
    announce(`Added ${g.display}`);
  }
  function clearSuggestions() { clearTimeout(suggestTimer); suggest = null; }
  function showSuggestions(top: Guess[], q: Question) {
    clearSuggestions();
    suggest = { top, question: q };
    suggestTimer = setTimeout(clearSuggestions, 6000);
  }
  const pick = (t: Guess, q: Question) => addGloss({ type: 'sign', label: t.label, display: t.display, question: q });

  // A live English preview under the chips, a short pause after the last change.
  let previewTimer: ReturnType<typeof setTimeout>;
  $effect(() => {
    const words = glosses.map((g) => (g.type === 'spell' ? g.label : g.display));
    const q = question, t = toneName;
    clearTimeout(previewTimer);
    if (!words.length) { preview = ''; return; }
    previewTimer = setTimeout(async () => {
      try { preview = (await postJSON<{ english: string }>('/api/sentence', { glosses: words, tone: t, question: q }, { quiet: true })).english; } catch { preview = ''; }
    }, 300);
  });

  function setMode(m: typeof mode) {
    mode = m;
    recognizer?.setMode(m);
    announce({ words: 'Words mode', letters: 'Letters mode', digits: 'Numbers mode' }[m]);
  }

  // ---- speaking ---------------------------------------------------------------------------------
  async function addMessage(m: Msg) {
    transcript.push(m);
    await tick();
    log?.scrollTo({ top: log.scrollHeight });
  }

  async function speakSentence() {
    if (!glosses.length) return;
    const tn = tracker?.takeUtteranceTone() ?? 'neutral';
    const words = glosses.map((g) => (g.type === 'spell' ? g.label : g.display));
    try {
      const { english } = await postJSON<{ english: string }>('/api/sentence', { glosses: words, tone: tn, question });
      addMessage({ who: 'you', english, detail: `${glosses.map((g) => g.display).join(' ')} · ${(TONE as any)[tn]?.[1] ?? 'Neutral'}`, tone: tn });
      speak(english, tn);
      announce(english);
      glosses = [];
    } catch { /* api.ts already showed the error */ }
  }

  // ---- the other person: typing or speaking, answered with signing ------------------------------
  async function playSigns(tokens: any[], msg: number) {
    await figure.play(tokens, { onWord: (_w, i) => (spoken = { msg, part: i }) });
    spoken = null;
  }

  async function sendReply(text: string) {
    text = text.trim();
    if (!text) return;
    try {
      const { tokens } = await postJSON<{ tokens: any[] }>('/api/gloss', { text });
      const parts = tokens.map((t) => (t.type === 'sign' ? t.label : t.text.split('').join('-')));
      await addMessage({ who: 'them', english: text, detail: parts.join(' '), tone: 'neutral', tokens, parts });
      playSigns(tokens, transcript.length - 1);
    } catch { /* api.ts already showed the error */ }
  }

  function submit(e: SubmitEvent) {
    e.preventDefault();
    const t = reply;
    reply = '';
    sendReply(t);
  }

  function mic() {
    if (rec) return rec.stop();
    rec = listen({
      onState: (on) => { listening = on; if (!on) rec = null; },
      onResult: ({ finals, interim }) => { reply = interim; finals.forEach((f) => { reply = ''; sendReply(f); }); },
    });
  }

  function exportChat() {
    if (!transcript.length) return toast('Nothing to save yet.');
    download('signbridge-conversation.txt', transcript.map((m) => `${m.who === 'you' ? 'Signer' : 'Reply'}: ${m.english}  (${m.detail})`).join('\n'));
  }

  // ---- keyboard ---------------------------------------------------------------------------------
  function keydown(e: KeyboardEvent) {
    const el = e.target as HTMLElement;
    if (el.matches?.('input, textarea, select') || e.metaKey || e.ctrlKey || e.altKey || document.querySelector('dialog[open]:not(#onboard)')) return;
    const k = e.key;
    if (k === 'i' || k === 'I') inspector = !inspector;
    else if (k === 'w' || k === 'W') setMode('words');
    else if (k === 'a' || k === 'A') setMode('letters');
    else if (k === 'n' || k === 'N') setMode('digits');
    else if (k === ' ') { e.preventDefault(); recognizer?.endSegment(); }
    else if (k === 'Enter') { e.preventDefault(); speakSentence(); }
    else if (k === 'Backspace') glosses.pop();
    else if (k === 'Escape') { if (suggest) clearSuggestions(); else glosses = []; }
    else if ('123'.includes(k) && suggest?.top[+k - 1]) pick(suggest.top[+k - 1], suggest.question);
  }

  const needsOnboarding = () => { try { return !localStorage.getItem('sb.onboarded'); } catch { return true; } };
  const onready = () => { if (needsOnboarding()) calibrating = 'first'; };
</script>

<svelte:window onkeydown={keydown} />
<svelte:head><title>Converse · SignBridge</title></svelte:head>

<h1 class="sr-only">Converse</h1>
<div class="grid-2">
  <section class="stack" aria-label="Your signing">
    <CameraStage bind:this={stage} bind:vision {busy} {onready}>
      {#snippet top()}
        <div class="stage__top">
          <div class="seg" role="radiogroup" aria-label="What you are signing">
            {#each [['words', 'Words'], ['letters', 'ABC'], ['digits', '123']] as [m, l] (m)}
              <button type="button" role="radio" aria-checked={mode === m} onclick={() => setMode(m as typeof mode)}>{l}</button>
            {/each}
          </div>
          <div class="row">
            <span class="tone" title="Tone read from your face"><ToneIcon /><span class="num">{TONE[toneName][1]}{toneConf ? ` ${Math.round(toneConf * 100)}%` : ""}</span></span>
            <button class="btn btn--icon btn--sm" type="button" aria-label="Model inspector" aria-pressed={inspector} title="Model inspector (I)" onclick={() => (inspector = !inspector)}><Gauge /></button>
            <button class="btn btn--icon btn--sm" type="button" aria-label="Recalibrate neutral face" onclick={() => (camera.state === 'on' ? (calibrating = 'again') : toast('Start the camera first.'))}><ScanFace /></button>
          </div>
        </div>
        {#if camera.state === 'on'}<span class="pill" data-busy={busy} aria-hidden="true">{status}</span>{/if}
        {#if inspector}
          <div class="inspector" aria-live="off">
            <div class="inspector__row"><span>Camera</span><b class="num">{fps.toFixed(0)} fps</b></div>
            <div class="inspector__row"><span>Last answer</span><b class="num">{last ? `${last.ms.toFixed(0)} ms` : 'none yet'}</b></div>
            <div class="inspector__row"><span>Frames in sign</span><b class="num">{last ? (last.frames ?? 'one frame') : 'none yet'}</b></div>
            <div>
              {#each last?.top ?? [] as t (t.label)}
                <div class="inspector__bar"><span>{t.display}</span><i class="meter" style:--p={t.prob}><i></i></i><b class="num">{Math.round(t.prob * 100)}%</b></div>
              {/each}
            </div>
          </div>
        {/if}
      {/snippet}
      {#snippet band()}<CandidateBand {cand} />{/snippet}
    </CameraStage>
    {#if calibrating && vision}
      <Onboarding {vision} neutralOnly={calibrating === 'again'} onfinish={() => (calibrating = null)} />
    {/if}
    {#if suggest}
      <div class="suggest" role="group" aria-label="Did you mean">
        <span class="label">Did you mean</span>
        {#each suggest.top as t, i (t.label)}
          <button class="chip chip--plain" type="button" onclick={() => pick(t, suggest!.question)}>{i + 1}  {t.display}</button>
        {/each}
      </div>
    {/if}
    <div class="panel">
      <div class="row row--between">
        <h2>Your sentence</h2>
        <span class="label" aria-live="polite">{QUESTION[question] ?? ''}</span>
      </div>
      <div class="row" aria-label="Signs so far">
        {#each glosses as g, i (g.id)}
          <span class="chip" class:chip--spell={g.type === 'spell'} animate:flip={{ duration: dur(200), easing: ease }} transition:fly={{ y: 8, duration: dur(200), easing: ease }}>
            {g.display}
            <button type="button" aria-label="Remove {g.display}" onclick={() => glosses.splice(i, 1)}><X /></button>
          </span>
        {:else}
          <span class="muted">Signs you make appear here.</span>
        {/each}
      </div>
      <p class="muted preview" aria-live="off">{preview ? `Preview: ${preview}` : ''}</p>
      <div class="row">
        <button class="btn btn--primary" type="button" disabled={!glosses.length} onclick={speakSentence}><Volume2 />Speak sentence</button>
        <button class="btn" type="button" disabled={!glosses.length} onclick={() => glosses.pop()}><Undo2 />Undo</button>
        <button class="btn btn--ghost" type="button" disabled={!glosses.length} onclick={() => (glosses = [])}><Trash2 />Clear</button>
        <button class="btn btn--ghost btn--icon" type="button" aria-label="Keyboard shortcuts" onclick={() => (shortcuts.open = true)}><Keyboard /></button>
      </div>
    </div>
  </section>

  <section class="stack" aria-label="Conversation">
    <div style="view-transition-name: figure"><Avatar bind:this={figure} /></div>
    <div class="panel">
      <div class="row row--between">
        <h2>Conversation</h2>
        <button class="btn btn--ghost btn--sm" type="button" onclick={exportChat}><Download />Save</button>
      </div>
      <div class="transcript" bind:this={log} role="log" aria-label="Messages">
        {#each transcript as m, i (i)}
          <div class="msg msg--{m.who}" in:fly={{ y: 10, duration: dur(250), easing: ease }}>
            <div class="msg__en">{m.english}</div>
            <div class="msg__meta">
              {#if m.parts}
                {#each m.parts as p, j (j)}<span class:now={spoken?.msg === i && spoken.part === j}>{p}</span>{/each}
              {:else}{m.detail}{/if}
              {#if m.who === 'you'}
                <button class="btn btn--ghost btn--icon btn--sm" type="button" aria-label="Replay voice" onclick={() => speak(m.english, m.tone)}><Volume2 /></button>
              {:else}
                <button class="btn btn--ghost btn--icon btn--sm" type="button" aria-label="Replay signs" onclick={() => playSigns(m.tokens!, i)}><Play /></button>
              {/if}
            </div>
          </div>
        {/each}
      </div>
      <form class="composer" onsubmit={submit}>
        <button class="btn btn--icon mic" type="button" aria-pressed={listening} aria-label="Speak a reply" disabled={!supported} title={supported ? undefined : 'Speech input needs Chrome or Edge'} onclick={mic}><Mic /></button>
        <input class="field" bind:value={reply} name="reply" placeholder="Type a reply" maxlength="500" autocomplete="off" aria-label="Type a reply" />
        <button class="btn btn--primary btn--icon" type="submit" aria-label="Send reply"><Send /></button>
      </form>
    </div>
  </section>
</div>

<style>
  .pill { position: absolute; left: 50%; top: var(--sc-3); translate: -50% 0; padding: 4px var(--sc-4); border-radius: var(--sc-r-pill); background: rgb(7 10 8 / 0.72); border: 1px solid var(--sc-hairline-strong); color: #eef1ea; font-size: var(--sc-t-sm); font-weight: 500; white-space: nowrap; }
  .pill[data-busy='true'] { animation: pulse 1s var(--sc-ease-out) infinite; }
  .preview { min-height: 1.6em; }
  .now { color: var(--sc-ink); background: color-mix(in oklab, var(--sc-accent) 40%, transparent); border-radius: 4px; padding: 0 4px; }
  .msg__meta { flex-wrap: wrap; }
</style>
