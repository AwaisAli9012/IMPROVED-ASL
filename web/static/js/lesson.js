// Lesson engine. Units run as: watch the avatar, try it on camera, repeat for each item, then a six question quiz.
// Intro units are pages; the sentence unit asks for whole sentences; "review" practises items that are due.
import { announce, config, el, fill, getJSON, icon, initShell, postJSON, refreshIcons, toast } from './api.js';
import { Avatar } from './avatar.js';
import { Recognizer } from './recognizer.js';
import { Vision, wireCamera } from './vision.js';

const $ = (id) => document.getElementById(id);
const unitId = $('lesson-root').dataset.unit;
const profile = await initShell({ requireProfile: true });
const cfg = await config();
const course = await getJSON(`/api/learn/${profile.id}`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const shuffle = (a) => a.map((x) => [Math.random(), x]).sort((p, q) => p[0] - q[0]).map(([, x]) => x);
const isHand = (item) => /^[A-Z0-9]$/.test(item);
const displayOf = Object.fromEntries(cfg.words.map((w) => [w.label, w.display]));
const show = (item) => displayOf[item] ?? item;

let unit = course.units.find((u) => u.id === unitId);
if (unitId === 'review') {
  const due = await getJSON(`/api/learn/${profile.id}/review`);
  unit = { id: 'review', title: 'Review', kind: 'review', state: 'open', items: due.map((d) => ({ item: d.item, display: d.display })) };
}
if (!unit || unit.state === 'locked') {
  toast('Finish the previous unit first.');
  setTimeout(() => { location.href = '/learn'; }, 1200);
  throw new Error('locked');
}
$('unit-title').textContent = unit.title;
document.title = `${unit.title} · SignBridge`;

// ---- shared parts --------------------------------------------------------------------------------
const avatar = new Avatar($('avatar'));
const vision = new Vision($('video'), $('overlay'));
let waiting = null; // set while a try is open: receives recognizer events
const recognizer = new Recognizer(cfg, {
  onGloss: (g) => waiting?.({ kind: 'gloss', label: g.label, display: g.display }),
  onChar: (c) => waiting?.({ kind: 'char', label: c, display: c }),
  onSuggest: (top) => waiting?.({ kind: 'suggest', top }),
  onBusy: (busy) => $('stage').setAttribute('aria-busy', busy),
  onCandidate: (c) => {
    $('cand-word').textContent = c?.display ?? '';
    $('cand-meter').dataset.state = c?.accept ? 'ok' : '';
    $('cand-meter').style.setProperty('--p', c?.prob ?? 0);
  },
});
vision.on((ev) => recognizer.push(ev));
wireCamera(vision);
$('speed').addEventListener('click', (e) => {
  if (!e.target.dataset.speed) return;
  avatar.setSpeed(+e.target.dataset.speed);
  document.querySelectorAll('#speed button').forEach((b) => b.setAttribute('aria-checked', b === e.target));
});
$('mirror').addEventListener('change', (e) => avatar.setMirror(e.target.checked));

const panel = $('panel');
const render = (...kids) => { fill(panel, ...kids); refreshIcons(); };
const button = (label, onclick, primary = true, ic) => el('button', { class: `btn${primary ? ' btn--primary' : ''}`, type: 'button', onclick }, ic ? icon(ic) : null, label);
const clicked = (...btns) => new Promise((resolve) => btns.forEach(([b, value]) => b.addEventListener('click', () => resolve(value), { once: true })));
const feedbackEl = () => el('p', { class: 'feedback', role: 'status' });
const say = (node, text, state = '') => { node.textContent = text; node.dataset.state = state; announce(text); };
const play = (item, loop = true) => avatar.play([{ type: 'sign', label: item }], { loop, onWord: (w) => { $('caption').textContent = w; } });

function setSteps(names, current) {
  $('steps').replaceChildren(...names.map((n) => el('li', n === current ? { 'aria-current': 'step' } : {}, n)));
}
function setProgress(fraction) { $('unit-bar').style.setProperty('--p', fraction); }
const record = (item, correct) => postJSON('/api/learn/attempt', { profile_id: profile.id, item, correct }, { quiet: true }).catch(() => {});

/** One try on the camera: resolves {ok} on the target, {ok:false, saw} on another sign, {ok:false} on timeout. */
function attempt(target, { timeout = 15000 } = {}) {
  recognizer.setMode(isHand(target) ? (/[0-9]/.test(target) ? 'digits' : 'letters') : 'words');
  return new Promise((resolve) => {
    const done = (r) => { clearTimeout(timer); waiting = null; recognizer.setMode('words'); resolve(r); };
    const timer = setTimeout(() => done({ ok: false }), timeout);
    waiting = (e) => {
      if (e.kind === 'suggest') return;
      if (e.label === target) done({ ok: true });
      else done({ ok: false, saw: e.display });
    };
  });
}

/** Keep trying one item until it is signed, or the learner chooses to skip after three misses. */
async function practise(item) {
  const fb = feedbackEl();
  let misses = 0;
  render(el('div', { class: 'big' }, show(item)), el('p', { class: 'muted' }, 'Sign it to the camera. You have about fifteen seconds.'), fb);
  for (;;) {
    say(fb, 'Waiting for your sign…');
    const r = await attempt(item);
    if (r.ok) { say(fb, 'Yes, that is it. +10 XP', 'ok'); await record(item, true); await sleep(900); return true; }
    misses++;
    say(fb, r.saw ? `That looked like “${r.saw}”. Watch it again.` : 'Time is up. Watch it again.', 'bad');
    play(item);
    if (misses >= 3) {
      const again = button('Try again', null, true), skip = button('Skip for now', null, false);
      render(el('div', { class: 'big' }, show(item)), fb, el('div', { class: 'row' }, again, skip));
      if (await clicked([again, 'again'], [skip, 'skip']) === 'skip') { await record(item, false); return false; }
      misses = 0;
      render(el('div', { class: 'big' }, show(item)), fb);
    }
  }
}

async function learnItem(item) {
  setSteps(['Watch', 'Try it', 'Quiz'], 'Watch');
  play(item);
  const go = button("I'm ready", null, true, 'arrow-right');
  render(el('div', { class: 'big' }, show(item)), el('p', { class: 'muted' }, 'Watch the sign. Slow it down if you need to, then try it yourself.'), el('div', { class: 'row' }, go));
  await clicked([go, 1]);
  setSteps(['Watch', 'Try it', 'Quiz'], 'Try it');
  await practise(item);
}

// ---- quiz -----------------------------------------------------------------------------------------
function pickDistractors(item, pool) {
  const others = shuffle([...new Set(pool)].filter((x) => x !== item));
  const filler = shuffle(isHand(item) ? [...cfg.LETTERS, ...cfg.DIGITS] : cfg.words.map((w) => w.label)).filter((x) => x !== item && !others.includes(x));
  return [...others, ...filler].slice(0, 3);
}

async function chooseQuestion(item, pool) {
  play(item);
  const options = shuffle([item, ...pickDistractors(item, pool)]);
  const fb = feedbackEl();
  const buttons = options.map((o) => button(show(o), null, false));
  render(el('h2', {}, 'What does this sign mean?'), el('div', { class: 'options' }, buttons), fb);
  const picked = await clicked(...buttons.map((b, i) => [b, options[i]]));
  buttons.forEach((b, i) => { b.disabled = true; if (options[i] === item) b.dataset.state = 'ok'; else if (options[i] === picked) b.dataset.state = 'bad'; });
  const ok = picked === item;
  say(fb, ok ? 'Correct' : `It was “${show(item)}”`, ok ? 'ok' : 'bad');
  await record(item, ok);
  await sleep(1100);
  return ok;
}

async function signQuestion(item) {
  avatar.stop();
  const fb = feedbackEl();
  render(el('h2', {}, 'Sign this word'), el('div', { class: 'big' }, show(item)), fb);
  say(fb, 'Waiting for your sign…');
  const r = await attempt(item);
  say(fb, r.ok ? 'Correct' : r.saw ? `That looked like “${r.saw}”` : 'Time is up', r.ok ? 'ok' : 'bad');
  await record(item, r.ok);
  await sleep(1100);
  return r.ok;
}

async function quiz(items) {
  setSteps(['Watch', 'Try it', 'Quiz'], 'Quiz');
  const picks = shuffle(items).slice(0, 6);
  while (picks.length < 6) picks.push(items[picks.length % items.length]);
  let score = 0;
  for (const [i, item] of picks.entries()) {
    setProgress((items.length + i / 6) / (items.length + 1));
    score += (i % 2 === 0 ? await chooseQuestion(item, items) : await signQuestion(item)) ? 1 : 0;
  }
  return score / 6;
}

// ---- finishing ---------------------------------------------------------------------------------------
async function finish(score) {
  avatar.stop();
  setProgress(1);
  const result = await postJSON('/api/learn/unit_pass', { profile_id: profile.id, unit_id: unit.id, score });
  const index = course.units.findIndex((u) => u.id === unit.id);
  const next = course.units[index + 1];
  if (!result.passed) {
    const again = button('Try again', () => location.reload(), true);
    render(el('h2', {}, `${Math.round(score * 100)}%. Nearly there`), el('p', { class: 'muted' }, 'You need 80% to unlock the next unit. A little more practice will do it.'), el('div', { class: 'row' }, again, el('a', { class: 'btn', href: '/learn' }, 'Back to the course')));
    return;
  }
  render(el('h2', {}, 'Unit complete'),
    el('p', { class: 'muted' }, `${Math.round(score * 100)}%${result.first_time ? ' · +50 XP' : ''}`),
    el('div', { class: 'row' }, next ? el('a', { class: 'btn btn--primary', href: `/learn/${next.id}` }, 'Next unit', icon('arrow-right')) : el('a', { class: 'btn btn--primary', href: '/certificate' }, icon('award'), 'Get your certificate'),
      el('a', { class: 'btn', href: '/learn' }, 'Back to the course')));
}

// ---- unit kinds ---------------------------------------------------------------------------------------
async function runIntro() {
  $('lesson-grid').hidden = true;
  setSteps(unit.pages.map((p) => p.title), unit.pages[0].title);
  for (let i = 0; i < unit.pages.length;) {
    const page = unit.pages[i];
    setSteps(unit.pages.map((p) => p.title), page.title);
    setProgress(i / unit.pages.length);
    const back = button('Back', null, false), next = button(i === unit.pages.length - 1 ? 'Finish' : 'Next', null, true);
    back.disabled = i === 0;
    render(el('h2', {}, page.title), el('p', { class: 'lede', style: 'font-size:var(--sc-t-lg);max-width:46ch' }, page.body), el('div', { class: 'row' }, back, next));
    i += (await clicked([back, -1], [next, 1]));
  }
  await finish(1);
}

async function runSentences() {
  const picks = shuffle(unit.sentences).slice(0, 5);
  let right = 0;
  for (const [i, s] of picks.entries()) {
    setProgress(i / picks.length);
    setSteps(['Watch', 'Try it', 'Quiz'], 'Try it');
    const fb = feedbackEl();
    const skip = button('Skip', null, false);
    render(el('h2', {}, 'Sign this sentence'), el('div', { class: 'big' }, s.english),
      el('p', { class: 'muted' }, `Signs, in order: ${s.gloss.map(show).join(', ')}`), fb, el('div', { class: 'row' }, skip));
    avatar.play(s.gloss.map((g) => ({ type: 'sign', label: g })), { loop: true, onWord: (w) => { $('caption').textContent = w; } });
    recognizer.setMode('words');
    const ok = await new Promise((resolve) => {
      let k = 0;
      say(fb, `Sign “${show(s.gloss[0])}”`);
      waiting = (e) => {
        if (e.kind === 'suggest') return;
        if (e.label !== s.gloss[k]) { k = 0; say(fb, `That looked like “${e.display}”. Start again with “${show(s.gloss[0])}”.`, 'bad'); return; }
        k++;
        if (k === s.gloss.length) { waiting = null; resolve(true); } else say(fb, `Good. Next: “${show(s.gloss[k])}”`, 'ok');
      };
      skip.addEventListener('click', () => { waiting = null; resolve(false); }, { once: true });
    });
    right += ok ? 1 : 0;
    say(fb, ok ? 'Sentence signed' : 'Skipped', ok ? 'ok' : '');
    await sleep(900);
  }
  await finish(right / picks.length);
}

async function main() {
  if (unit.kind === 'intro') return runIntro();
  if (unit.kind === 'sentence') return runSentences();
  const items = unit.items.map((i) => i.item);
  if (unit.kind === 'review') {
    if (!items.length) return render(el('h2', {}, 'Nothing due'), el('p', { class: 'muted' }, 'Come back tomorrow for a review.'), el('a', { class: 'btn btn--primary', href: '/learn' }, 'Back to the course'));
    for (const [i, item] of items.entries()) { setProgress(i / items.length); setSteps(['Review'], 'Review'); play(item); await practise(item); }
    avatar.stop();
    return render(el('h2', {}, 'Review done'), el('a', { class: 'btn btn--primary', href: '/learn' }, 'Back to the course'));
  }
  for (const [i, item] of items.entries()) { setProgress(i / (items.length + 1)); await learnItem(item); }
  await finish(await quiz(items));
}
main().catch((e) => { if (e.message !== 'locked') console.error(e); });
