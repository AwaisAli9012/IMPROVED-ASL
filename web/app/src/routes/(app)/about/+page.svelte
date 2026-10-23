<script lang="ts">
  import { onMount } from 'svelte';
  import { getJSON, type Cfg } from '$lib/api';
  import { reveal } from '$lib/actions';
  import CountUp from '$lib/components/CountUp.svelte';

  const pct = new Intl.NumberFormat('en', { style: 'percent', minimumFractionDigits: 1, maximumFractionDigits: 1 });
  let cfg = $state<Cfg | null>(null);
  let units = $state<unknown[] | null>(null);
  let reports = $state<Record<string, any> | null>(null);
  let health = $state<any>(null);

  onMount(async () => {
    const q = <T,>(url: string) => getJSON<T>(url, { quiet: true }).catch(() => null);
    [cfg, units, reports, health] = await Promise.all([q<Cfg>('/api/config'), q<unknown[]>('/api/curriculum'), q<Record<string, any>>('/api/reports'), q<any>('/api/health')]);
  });

  const rows = $derived([['words', 'Words'], ['static', 'Letters and numbers'], ['emotion', 'Tone from the face']].filter(([k]) => reports?.[k]));
  const flow = [
    ['Camera', 'Your browser reads hands, body and face points. The video never leaves the page.'],
    ['Points', '89 numbers per frame go to a server on this computer.'],
    ['Models', 'A sequence model reads whole signs. A second one reads fingerspelling. A third reads expression.'],
    ['Sentence', 'ASL order becomes English with the right tone and question form.'],
    ['Voice and signing', 'The browser speaks it. Replies are signed back by the figure.'],
  ];
</script>

<svelte:head><title>About · SignBridge</title></svelte:head>

<article class="stack about" aria-label="About SignBridge">
  <div class="stack">
    <h1>About SignBridge</h1>
    <p class="lede">A two-way American Sign Language tool: sign to a camera and get a sentence and a voice, type or speak and get signing back, and learn the language along the way. It runs on one computer, on a CPU.</p>
  </div>

  <section class="stack"><h2>How it works</h2>
    <ol class="flow" aria-label="Pipeline">
      {#each flow as [name, text], i (name)}
        <li class="reveal" use:reveal={i * 90}><b>{name}</b><span>{text}</span></li>
      {/each}
    </ol>
  </section>

  <section class="stack"><h2>What it knows</h2>
    <div class="kpis">
      <div class="kpi"><b>{#if cfg}<CountUp value={cfg.words.length} />{:else}…{/if}</b><span class="label">Words</span></div>
      <div class="kpi"><b>{#if cfg}<CountUp value={cfg.LETTERS.length} />{:else}…{/if}</b><span class="label">Letters</span></div>
      <div class="kpi"><b>{#if cfg}<CountUp value={cfg.DIGITS.length} />{:else}…{/if}</b><span class="label">Numbers</span></div>
      <div class="kpi"><b>{#if units}<CountUp value={units.length} />{:else}…{/if}</b><span class="label">Course units</span></div>
    </div>
    <p class="muted">Anything outside the vocabulary is fingerspelled letter by letter. Numbers 0 to 4 and 6 to 9 are supported: no open dataset has a signed 5 yet.</p>
  </section>

  <section class="stack"><h2>How well it works</h2>
    <p class="muted">Each model is tested on data it never trained on. For words that means signers it never saw.</p>
    <div class="table-wrap"><table class="table num"><thead><tr><th scope="col">Task</th><th scope="col">Tested on</th><th scope="col">Accuracy</th><th scope="col">Macro F1</th><th scope="col">Answers in</th></tr></thead>
      <tbody>
        {#each rows as [k, name] (k)}
          {@const r = reports![k]}
          <tr><td>{name}</td><td>{r.split.test_signers ? `${r.split.n_test} clips, ${r.split.test_signers.length} unseen signers` : `${r.split.n_test} samples`}</td>
            <td>{pct.format(r.accuracy)}</td><td>{pct.format(r.f1_macro)}</td><td>{r.latency_ms_p95.toFixed(1)} ms</td></tr>
        {/each}
      </tbody></table></div>
    <p><a href="/metrics">See every metric, the confusion matrices and per-sign scores</a></p>
  </section>

  <section class="stack"><h2>Privacy</h2>
    <ul class="stack">
      <li>Your video is processed inside the browser tab and is never recorded or uploaded.</li>
      <li>Only landmark numbers (points on your hands, body and lips) are sent, to the server running next to the app.</li>
      <li>Your progress is stored in a small database on the computer running SignBridge. Settings and your name are stored in your browser. There are no accounts and no analytics.</li>
      <li>You can download or permanently delete your progress in <a href="/settings">Settings</a>.</li>
      {#if cfg?.gemini}<li>A Gemini key is configured, so the signs you string together (as text) are sent to Google to make the sentence read naturally. No video or points are sent. Without a key, a simple offline rule builds the sentence.</li>{/if}
    </ul>
  </section>

  <section class="stack"><h2>Ethics and fairness</h2>
    <ul class="stack">
      <li>SignBridge is a learning and communication aid. It does not replace a qualified interpreter, and it should not be used for medical, legal or emergency communication on its own.</li>
      <li>It is trained on webcam recordings from 52 volunteer signers in the ASL Citizen dataset, used under its research licence. Results on other people, cameras, lighting and signing styles will differ. We have not audited accuracy across skin tones.</li>
      <li>It covers American Sign Language only, and individual signs only: not the facial grammar and spatial grammar of fluent signing.</li>
      <li>Tone read from your face is a hint for the wording and voice of a sentence. It is not a judgement of how anyone feels.</li>
      <li>Sign language belongs to the Deaf community. Learners should treat this as a starting point and learn from Deaf teachers.</li>
    </ul>
  </section>

  <section class="stack"><h2>Accessibility</h2>
    <p class="muted">Keyboard-only use, screen-reader announcements for signs and sentences, a high-contrast theme, adjustable text size, a reduce-motion option, live captions for speech, and a phrasebook for quick needs. All are in <a href="/settings">Settings</a>.</p>
  </section>

  <section class="stack"><h2>Keyboard shortcuts</h2>
    <div class="table-wrap"><table class="table"><tbody>
      <tr><td><b>Ctrl K · /</b></td><td>Search pages, signs and phrases</td></tr>
      <tr><td><b>?</b></td><td>Show the shortcuts for the page you are on</td></tr>
      <tr><td><b>W · A · N</b></td><td>Converse: words, letters, numbers mode</td></tr>
      <tr><td><b>Space</b></td><td>Converse: end the sign you are making now</td></tr>
      <tr><td><b>1 · 2 · 3</b></td><td>Converse: pick a suggestion</td></tr>
      <tr><td><b>Enter</b></td><td>Converse: speak the sentence</td></tr>
      <tr><td><b>Backspace · Esc</b></td><td>Converse: remove the last sign · clear</td></tr>
      <tr><td><b>I</b></td><td>Converse: show the model inspector</td></tr>
    </tbody></table></div>
  </section>

  <section class="stack"><h2>Credits and licences</h2>
    <ul class="stack">
      <li>ASL Citizen (Microsoft Research), used under the Microsoft Research License: non-commercial research only. The data and signing animations made from it are not redistributed.</li>
      <li>ASL alphabet images (Hugging Face, <i>Marxulia/asl_sign_languages_alphabets_v03</i>) and FER2013 faces (Hugging Face, <i>Aaryan333/fer2013_train_publicTest_privateTest</i>).</li>
      <li>MediaPipe (Google, Apache 2.0) for hand, pose and face landmarks. Lucide icons (ISC). Archivo, Geist and Geist Mono fonts (SIL Open Font Licence).</li>
    </ul>
    {#if health}<p class="muted">Status: {health.status}. Models loaded: {Object.entries(health.models).map(([k, v]) => `${k} ${v ? 'yes' : 'no'}`).join(', ')}.</p>{/if}
  </section>
</article>
