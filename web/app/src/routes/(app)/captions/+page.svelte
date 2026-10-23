<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { Download, Maximize, Mic, Minus, Plus, Trash2 } from '@lucide/svelte';
  import { download, listen, postJSON, speechSupported } from '$lib/api';
  import Avatar from '$lib/components/Avatar.svelte';
  import { prefs, setPref } from '$lib/stores/prefs.svelte';
  import { toast } from '$lib/stores/toasts.svelte';

  let log = $state<{ text: string; at: string }[]>([]);
  let interim = $state('');
  let listening = $state(false);
  let signIt = $state(false);
  let supported = $state(true);
  let wanted = false;
  let rec: { stop(): void } | null = null;
  let figure: Avatar | undefined = $state();
  let box: HTMLElement;

  const cap = $derived(`${1.1 + prefs.captionSize * 0.55}rem`);
  onMount(() => { supported = speechSupported(); return () => { wanted = false; rec?.stop(); }; });

  function start() {
    wanted = true;
    rec = listen({
      continuous: true,
      onState: (on) => {
        listening = on;
        if (!on && wanted) setTimeout(start, 300); // the browser ends a session after a pause: keep going until Stop
      },
      onResult: async ({ finals, interim: i }) => {
        for (const text of finals) {
          log.push({ text, at: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) });
          if (signIt && figure) postJSON('/api/gloss', { text }, { quiet: true }).then(({ tokens }) => figure!.play(tokens)).catch(() => {});
        }
        interim = i;
        await tick();
        window.scrollTo({ top: document.body.scrollHeight });
      },
    });
  }

  const toggle = () => { if (wanted) { wanted = false; rec?.stop(); } else start(); };
  const save = () => (log.length ? download('captions.txt', log.map((l) => `${l.at}  ${l.text}`).join('\n')) : toast('Nothing to save yet.'));
</script>

<svelte:head><title>Live captions · SignBridge</title></svelte:head>

<section class="stack" aria-label="Live captions">
  <div class="row row--between">
    <div class="stack"><h1>Live captions</h1><p class="muted">Turns what people say into large text as they speak. Everything stays in this browser tab.</p></div>
    <div class="row">
      <button class="btn btn--primary" type="button" aria-pressed={listening} disabled={!supported} onclick={toggle}><Mic /><span>{listening ? 'Stop listening' : 'Start listening'}</span></button>
      <button class="btn btn--icon" type="button" aria-label="Smaller text" onclick={() => setPref('captionSize', Math.max(0, prefs.captionSize - 1))}><Minus /></button>
      <button class="btn btn--icon" type="button" aria-label="Larger text" onclick={() => setPref('captionSize', Math.min(8, prefs.captionSize + 1))}><Plus /></button>
      <button class="btn btn--icon" type="button" aria-label="Full screen" onclick={() => box.requestFullscreen?.()}><Maximize /></button>
      <label class="row" style="gap: var(--sc-2)"><input type="checkbox" bind:checked={signIt} /> <span>Sign each sentence</span></label>
      <button class="btn" type="button" onclick={save}><Download />Save</button>
      <button class="btn btn--ghost" type="button" onclick={() => (log = [])}><Trash2 />Clear</button>
    </div>
  </div>
  {#if !supported}<p class="muted">Live captions need speech recognition, which works in Chrome and Edge.</p>{/if}
  <div class="captions" bind:this={box} style:--cap={cap} role="log" aria-live="polite" aria-label="Captions">
    {#each log as l, i (i)}
      <p class="captions__line" class:old={i < log.length - 1}><span class="label num">{l.at}</span> {l.text}</p>
    {/each}
    {#if interim}<p class="captions__line captions__line--live">{interim}</p>{/if}
  </div>
  {#if signIt}
    <div class="panel"><Avatar bind:this={figure} controls={false} /></div>
  {/if}
</section>

<style>
  .old { color: var(--sc-ink-soft); }
  .captions:fullscreen { min-height: 100vh; padding: var(--sc-7); }
  .captions__line { animation: line-in 0.3s var(--sc-ease-out) both; }
  @keyframes line-in { from { opacity: 0; transform: translateY(8px); } }
</style>
