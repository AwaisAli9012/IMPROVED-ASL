<script lang="ts">
  import { onMount } from 'svelte';
  import { startLanding } from '$lib/landing/landing';

  let ready = $state(false);

  /** The engine and its styles load only here, so they never touch the app pages. */
  function load(tag: 'link' | 'script', url: string, first = false) {
    return new Promise<HTMLElement>((resolve, reject) => {
      const n = document.createElement(tag);
      if (n instanceof HTMLLinkElement) { n.rel = 'stylesheet'; n.href = url; } else n.src = url;
      n.onload = () => resolve(n);
      n.onerror = () => reject(new Error(url));
      if (first) document.head.prepend(n); else document.head.append(n); // the engine's tokens must come before the app's brand tokens
    });
  }

  onMount(() => {
    document.body.classList.add('landing');
    let stop: (() => void) | undefined;
    (async () => {
      await Promise.all([load('link', '/landing/scrollcraft.css', true), load('link', '/landing/landing.css')]);
      await load('script', '/landing/scrollcraft.js');
      ready = true;
      stop = await startLanding();
    })().catch((e) => console.error(e));
    return () => { document.body.classList.remove('landing'); stop?.(); };
  });
</script>

<svelte:head><title>SignBridge · sign language, both ways</title></svelte:head>

<!-- shortcut: the scroll engine has no teardown, so every link off this page is a full page load. -->
<div data-sveltekit-reload style:visibility={ready ? 'visible' : 'hidden'}>
<span data-sc-progress aria-hidden="true"></span>
<div class="sc-grain" aria-hidden="true"></div>

<header class="chrome">
  <a class="chrome__mark" href="/">SignBridge</a>
  <nav aria-label="Product pages">
    <a href="/app">Converse</a>
    <a href="/learn">Learn</a>
    <a href="/dictionary">Dictionary</a>
    <a href="/metrics">Metrics</a>
  </nav>
  <span class="chrome__status"><i class="dot" data-state="on"></i>Runs on this computer</span>
</header>

<main id="top">
  <h1 class="sr-only">SignBridge: sign language, both ways</h1>

  <!-- 1 · Your name, signed: the surface, already in a state -->
  <section id="act-name" data-sc-act="pin" data-sc-span="2.4" data-sc-drift="#0d110f">
    <div data-sc-stage class="stage-center">
      <canvas class="constellation" id="field" aria-hidden="true"></canvas>
      <i class="ring ring--a" data-sc-parallax="-0.22" aria-hidden="true"></i>
      <i class="ring ring--b" data-sc-parallax="0.3" aria-hidden="true"></i>
      <div class="surface" data-sc-tilt="4">
        <div class="surface__bar"><span class="label">Fingerspelling</span><span class="label num" id="spell-count"></span></div>
        <div class="avatar avatar--tall"><!-- svelte-ignore a11y_no_interactive_element_to_noninteractive_role --><canvas id="name-avatar" role="img" aria-label="Signing figure spelling the name you typed"></canvas></div>
        <ol class="tiles" id="tiles" aria-live="polite" aria-label="Letters being signed"></ol>
        <form class="composer" id="name-form">
          <label class="sr-only" for="name-input">Your name</label>
          <input class="field" id="name-input" value="Ada" maxlength="14" placeholder="Type your name" autocomplete="given-name" spellcheck="false">
          <button class="btn btn--primary" type="submit">Sign it</button>
        </form>
        <p class="hint" id="name-hint">Type a name. Scroll to step through its letters.</p>
      </div>
    </div>
  </section>

  <!-- 2 · How it reads you: real landmarks assemble, then real measured numbers -->
  <section id="act-reads" data-sc-act="pin" data-sc-span="3.6" data-sc-drift="#0d110f">
    <div data-sc-stage class="stage-center">
      <i class="ring ring--c" data-sc-parallax="0.25" aria-hidden="true"></i>
      <div class="reads">
        <figure class="points" id="points-box">
          <!-- svelte-ignore a11y_no_interactive_element_to_noninteractive_role -->
          <canvas id="points" role="img" aria-label="The points SignBridge reads from a sign, flying together into hands, arms and lips"></canvas>
          <figcaption class="label num" id="points-caption"></figcaption>
        </figure>
        <ol class="readout" id="readout">
          <li data-sc-cue="0 0.30 0">
            <span class="label">1 · Camera</span>
            <p class="readout__line">Your video stays on this computer. Only points leave the browser, to a server running next to it.</p>
          </li>
          <li data-sc-cue="0.24 0.54">
            <span class="label">2 · Points</span>
            <p class="readout__line"><b class="num" id="n-points">89</b> points in every frame: both hands, your shoulders and arms, and your lips.</p>
          </li>
          <li data-sc-cue="0.48 0.78">
            <span class="label">3 · Model</span>
            <p class="readout__line">One network reads the movement of a whole sign, not a single frame. It knows <b class="num" id="n-words">280</b> words, the alphabet and the numbers.</p>
          </li>
          <li data-sc-cue="0.72 1" id="measured" hidden>
            <span class="label">4 · Measured</span>
            <p class="readout__line">On <b class="num" id="n-signers">11</b> signers it never saw while learning:</p>
            <dl class="figures">
              <div><dt>Word accuracy</dt><dd class="num"><span id="f-words" data-sc-count="0 0" data-sc-count-at="0.74 0.96">0</span>%</dd></div>
              <div><dt>Letters, F1</dt><dd class="num"><span id="f-static" data-sc-count="0 0" data-sc-count-at="0.74 0.96">0</span>%</dd></div>
              <div><dt>Answers in</dt><dd class="num"><span id="f-speed" data-sc-count="0 0" data-sc-count-at="0.74 0.96">0</span> ms</dd></div>
            </dl>
          </li>
        </ol>
      </div>
      <ol class="rail" id="rail" aria-hidden="true"><li></li><li></li><li></li><li></li></ol>
    </div>
  </section>

  <!-- 3 · The conversation: the peak. Scripted sample, real signing. -->
  <section id="act-talk" data-sc-act="pin" data-sc-span="5.6" data-sc-drift="#101612">
    <div data-sc-stage class="stage-center">
      <i class="ring ring--a" data-sc-parallax="0.18" aria-hidden="true"></i>
      <div class="surface surface--wide">
        <div class="surface__bar"><span class="label">Conversation</span><span class="label num" id="talk-step">Scripted sample. The app does this live.</span></div>
        <div class="talk">
          <div class="stack">
            <div class="avatar" id="talk-frame"><!-- svelte-ignore a11y_no_interactive_element_to_noninteractive_role --><canvas id="talk-avatar" role="img" aria-label="Signing figure replying"></canvas></div>
            <p class="caption" id="talk-caption" aria-hidden="true"></p>
          </div>
          <div class="transcript" id="talk-log" aria-live="polite" data-lenis-prevent></div>
        </div>
      </div>
    </div>
  </section>

  <!-- 3b · The range: every topic as a column, tap a word to see it signed -->
  <section id="act-range" data-sc-act="pan" data-sc-span="8" data-sc-drift="#0d110f">
    <div data-sc-stage class="stage-center range">
      <div class="range__fig">
        <div class="avatar"><!-- svelte-ignore a11y_no_interactive_element_to_noninteractive_role --><canvas id="range-avatar" role="img" aria-label="Signing figure showing the chosen word"></canvas></div>
        <p class="caption" id="range-caption" aria-live="polite"></p>
      </div>
      <div class="rail" data-sc-pan="0.04" id="range-rail"></div>
    </div>
  </section>

  <!-- 4 · What is inside: a plain list of the real pages -->
  <section id="act-inside" class="sc-section inside" data-sc-act="flow" data-sc-drift="#0d110f">
    <div class="sc-wrap">
      <ul class="palette" data-sc-in data-sc-stagger="70" aria-label="Inside SignBridge">
        <li><a href="/app"><span class="palette__name">Converse</span><span class="muted">Sign to text and voice. Type or speak to get signing back.</span><span class="palette__go" aria-hidden="true">→</span></a></li>
        <li><a href="/learn"><span class="palette__name">Learn</span><span class="muted" id="inside-learn">A full course from camera setup to whole sentences.</span><span class="palette__go" aria-hidden="true">→</span></a></li>
        <li><a href="/phrasebook"><span class="palette__name">Phrases</span><span class="muted">Tap a phrase to see it signed and hear it, or add your own.</span><span class="palette__go" aria-hidden="true">→</span></a></li>
        <li><a href="/dictionary"><span class="palette__name">Dictionary</span><span class="muted" id="inside-dict">Every sign, searchable, played by the signing figure.</span><span class="palette__go" aria-hidden="true">→</span></a></li>
        <li><a href="/captions"><span class="palette__name">Live captions</span><span class="muted">Speech turned into large text as people talk, signed back if you like.</span><span class="palette__go" aria-hidden="true">→</span></a></li>
        <li><a href="/progress"><span class="palette__name">Progress</span><span class="muted">A heatmap of every sign, your weak spots and badges.</span><span class="palette__go" aria-hidden="true">→</span></a></li>
        <li><a href="/metrics"><span class="palette__name">Metrics</span><span class="muted">Every accuracy number, measured on held-out signers, with the confusion matrix.</span><span class="palette__go" aria-hidden="true">→</span></a></li>
      </ul>
    </div>
  </section>

  <!-- 5 · Open: an actual input, last on the page -->
  <section id="act-open" data-sc-act="pin" data-sc-span="1.5" data-sc-drift="#0d110f">
    <div data-sc-stage class="stage-center stage-close">
      <i class="ring ring--b" data-sc-parallax="-0.2" aria-hidden="true"></i>
      <form class="surface surface--close" id="open-form" data-sc-cue="0 1 0 0">
        <label class="label" for="open-name">What should we call you?</label>
        <div class="composer">
          <input class="field" id="open-name" maxlength="40" placeholder="Your name" autocomplete="given-name" required>
          <button class="btn btn--primary" type="submit">Open SignBridge</button>
        </div>
      </form>
      <footer class="foot">
        <span>A final-year project. Signing data: ASL Citizen, Microsoft Research License, non-commercial research use. Accuracy is measured on signers held out of training.</span>
        <a href="/metrics">See the numbers</a>
      </footer>
    </div>
  </section>

</main>

</div>
