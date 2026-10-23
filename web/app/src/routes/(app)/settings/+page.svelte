<script lang="ts">
  import { onMount } from 'svelte';
  import { Camera, Download, RotateCcw, Trash2, Volume2 } from '@lucide/svelte';
  import { deleteJSON, download, getJSON, speak } from '$lib/api';
  import { prefs, setPref, type Prefs } from '$lib/stores/prefs.svelte';
  import { profile } from '$lib/stores/profile.svelte';
  import { toast } from '$lib/stores/toasts.svelte';

  const themes = [['dark', 'Dark'], ['light', 'Light'], ['contrast', 'High contrast']] as const;
  const sizes = [['s', 'Small'], ['m', 'Medium'], ['l', 'Large'], ['xl', 'Extra large']] as const;
  const speeds = [[0.5, 'Slow (0.5×)'], [1, 'Normal (1×)']] as const;

  let voices = $state<SpeechSynthesisVoice[]>([]);
  let cams = $state<{ id: string; label: string }[]>([]);
  let camNote = $state('Camera names appear after you allow access once. The video never leaves this computer.');
  const hasSpeech = typeof window !== 'undefined' && 'speechSynthesis' in window;

  onMount(() => {
    if (!hasSpeech) return;
    const load = () => (voices = speechSynthesis.getVoices().filter((v) => v.lang.startsWith('en')));
    load();
    speechSynthesis.addEventListener('voiceschanged', load);
    return () => speechSynthesis.removeEventListener('voiceschanged', load);
  });

  function pickTheme(t: Prefs['theme']) {
    if (t === prefs.theme) return;
    if (document.startViewTransition && !prefs.reduceMotion) document.startViewTransition(() => setPref('theme', t));
    else setPref('theme', t);
  }

  async function listCameras() {
    try {
      (await navigator.mediaDevices.getUserMedia({ video: true })).getTracks().forEach((t) => t.stop());
      const found = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
      cams = found.map((d, i) => ({ id: d.deviceId, label: d.label || `Camera ${i + 1}` }));
      camNote = `${found.length} camera${found.length === 1 ? '' : 's'} found.`;
    } catch { toast('Camera access was blocked.', 'error'); }
  }

  async function exportData() {
    const me = profile.current;
    if (!me) return toast('Open Learn and create a profile first.');
    download(`signbridge-progress-${me.name}.json`, JSON.stringify(await getJSON(`/api/progress/${me.id}`), null, 2), 'application/json');
  }

  function resetPrefs() {
    try { localStorage.removeItem('sb.prefs'); } catch { /* nothing to reset */ }
    location.reload();
  }

  async function deleteProfile() {
    const me = profile.current;
    if (!me) return toast('There is no profile on this browser.');
    if (!confirm(`Delete "${me.name}" and all of their progress? This cannot be undone.`)) return;
    await deleteJSON(`/api/profiles/${me.id}`);
    try { localStorage.removeItem('sb.profile'); localStorage.removeItem('sb.baseline'); } catch { /* ignore */ }
    toast('Profile deleted.');
    setTimeout(() => location.assign('/'), 900);
  }
</script>

<svelte:head><title>Settings · SignBridge</title></svelte:head>

<section class="stack" aria-label="Settings">
  <div class="stack"><h1>Settings</h1><p class="muted">Saved on this computer. Nothing here is sent anywhere.</p></div>

  <div class="panel">
    <h2>Appearance</h2>
    <div class="field-row"><span id="theme-label" class="label">Theme</span>
      <div class="choice" role="radiogroup" aria-labelledby="theme-label">
        {#each themes as [v, l] (v)}<button class="chip chip--plain" type="button" role="radio" aria-checked={prefs.theme === v} onclick={() => pickTheme(v)}>{l}</button>{/each}
      </div></div>
    <div class="field-row"><span id="size-label" class="label">Text size</span>
      <div class="choice" role="radiogroup" aria-labelledby="size-label">
        {#each sizes as [v, l] (v)}<button class="chip chip--plain" type="button" role="radio" aria-checked={prefs.textSize === v} onclick={() => setPref('textSize', v)}>{l}</button>{/each}
      </div></div>
    <label class="row"><input type="checkbox" checked={prefs.reduceMotion} onchange={(e) => setPref('reduceMotion', e.currentTarget.checked)} /> <span>Reduce motion (turns off animations and transitions)</span></label>
  </div>

  <div class="panel">
    <h2>Voice</h2>
    <div class="field-row"><label for="voice">Spoken voice</label>
      <select class="field" id="voice" disabled={!hasSpeech} value={prefs.voice} onchange={(e) => setPref('voice', e.currentTarget.value)}>
        <option value="">Browser default</option>
        {#each voices as v (v.voiceURI)}<option value={v.voiceURI}>{v.name} ({v.lang})</option>{/each}
      </select></div>
    <div class="field-row"><label for="rate">Speaking speed <span class="num muted">{prefs.rate.toFixed(1)}×</span></label>
      <input type="range" id="rate" min="0.6" max="1.4" step="0.1" value={prefs.rate} oninput={(e) => setPref('rate', +e.currentTarget.value)} /></div>
    <div class="row"><button class="btn" type="button" onclick={() => speak('Hello, this is SignBridge.')}><Volume2 />Test voice</button></div>
    {#if !hasSpeech}<p class="muted">This browser has no speech output.</p>{/if}
  </div>

  <div class="panel">
    <h2>Camera</h2>
    <div class="field-row"><label for="camera">Camera to use</label>
      <select class="field" id="camera" value={prefs.camera} onchange={(e) => setPref('camera', e.currentTarget.value)}>
        <option value="">Default camera</option>
        {#if prefs.camera && !cams.some((c) => c.id === prefs.camera)}<option value={prefs.camera}>Your chosen camera</option>{/if}
        {#each cams as c (c.id)}<option value={c.id}>{c.label}</option>{/each}
      </select></div>
    <div class="row"><button class="btn" type="button" onclick={listCameras}><Camera />Show my cameras</button></div>
    <p class="muted">{camNote}</p>
  </div>

  <div class="panel">
    <h2>Signing figure</h2>
    <div class="field-row"><span id="speed-label" class="label">Default speed</span>
      <div class="choice" role="radiogroup" aria-labelledby="speed-label">
        {#each speeds as [v, l] (v)}<button class="chip chip--plain" type="button" role="radio" aria-checked={prefs.avatarSpeed === v} onclick={() => setPref('avatarSpeed', v)}>{l}</button>{/each}
      </div></div>
    <label class="row"><input type="checkbox" checked={prefs.mirror} onchange={(e) => setPref('mirror', e.currentTarget.checked)} /> <span>Mirror me by default (shows signs as you would make them)</span></label>
  </div>

  <div class="panel">
    <h2>Your data</h2>
    <p class="muted">Progress is stored in a small database on the computer running SignBridge.</p>
    <div class="row">
      <button class="btn" type="button" onclick={exportData}><Download />Download my progress</button>
      <button class="btn" type="button" onclick={resetPrefs}><RotateCcw />Reset settings</button>
      <button class="btn" type="button" onclick={deleteProfile}><Trash2 />Delete my profile and progress</button>
    </div>
  </div>
</section>
