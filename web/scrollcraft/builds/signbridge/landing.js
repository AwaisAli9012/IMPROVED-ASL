// Landing behaviour. The signature move lives here: the visitor's name, fingerspelled by the real signing
// figure, stepped by the scroll wheel, and returned in the conversation. Every number comes from the API.
import { Avatar } from '/static/js/avatar.js';

const $ = (id) => document.getElementById(id);
const sign = (label) => ({ type: 'sign', label });
const get = (url) => fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(new Error(url)))).catch(() => null);
const p = (id) => parseFloat($(id).style.getPropertyValue('--sc-p')) || 0; // progress the engine publishes on each act

// ---- real numbers, written before the engine mounts so the counters have their targets -------------
const [cfg, reports, units] = await Promise.all([get('/api/config'), get('/api/reports'), get('/api/curriculum')]);
if (cfg) {
  $('n-points').textContent = cfg.POSE_IDX.length + cfg.LIP_IDX.length + 42;
  $('n-words').textContent = cfg.words.length;
  $('inside-dict').textContent = `${cfg.words.length + cfg.LETTERS.length + cfg.DIGITS.length} signs, searchable, each played by the signing figure.`;
}
if (units) $('inside-learn').textContent = `A full course in ${units.length} units, from camera setup to whole sentences.`;
const words = reports?.words, letters = reports?.static;
if (words && letters) {
  const counter = (id, value) => $(id).setAttribute('data-sc-count', `0 ${value}`);
  counter('f-words', (words.accuracy * 100).toFixed(1));
  counter('f-static', (letters.f1_macro * 100).toFixed(1));
  counter('f-speed', Math.max(1, Math.round(words.latency_ms_p95)));
  $('n-signers').textContent = words.split.test_signers.length;
  $('measured').hidden = false;
} else {
  $('readout').children[2].setAttribute('data-sc-cue', '0.48 1'); // no report yet: the last cue holds through the act
}
ScrollCraft.mount(document.body);

// ---- act 1: your name, signed ------------------------------------------------------------------------
const clean = (s) => s.toUpperCase().replace(/[^A-Z]/g, '').slice(0, 14);
const title = (s) => s[0] + s.slice(1).toLowerCase();
let name = clean($('name-input').value) || 'ADA';
let mode = 'play';
const nameAvatar = new Avatar($('name-avatar'));

function spelled(i = -1) {
  $('spelled').replaceChildren(...[...name].map((c, k) => Object.assign(document.createElement('span'), { textContent: c, ariaCurrent: String(k === i) })));
}
function play() {
  mode = 'play';
  $('name-hint').textContent = 'Type a name. Scroll to step through its letters.';
  nameAvatar.play([{ type: 'spell', text: name }], { loop: true, onWord: (_, i) => spelled(i) });
}
$('name-form').addEventListener('submit', (e) => {
  e.preventDefault();
  name = clean($('name-input').value) || name;
  $('open-name').value = title(name);
  play();
});
$('name-input').addEventListener('input', () => { const n = clean($('name-input').value); if (n) { name = n; spelled(); play(); } });
spelled();
play();
$('open-name').value = title(name);

// ---- act 3: the conversation (scripted sample, real signing) --------------------------------------------
const talkAvatar = new Avatar($('talk-avatar'));
const SCRIPT = [
  { who: 'you', en: 'Are you hungry?', meta: 'you hungry · yes/no question · happy' },
  { who: 'them', en: 'Yes, I want to eat.', tokens: () => ['yes', 'me', 'want', 'eat'].map(sign) },
  { who: 'you', en: 'I am going home.', meta: 'me go home · neutral' },
  { who: 'them', en: 'See you tomorrow.', tokens: () => ['see', 'you', 'tomorrow'].map(sign) },
  { who: 'you', en: 'What is your name?', meta: 'what you name · what question · neutral' },
  { who: 'them', en: () => `My name is ${title(name)}.`, tokens: () => [sign('my'), sign('name'), { type: 'spell', text: name }] },
];
const AT = [0.18, 0.32, 0.46, 0.6, 0.74, 0.86]; // the first 18% is a held surface, by design
let shown = 0;

function renderTalk(k) {
  const log = $('talk-log');
  log.replaceChildren(...SCRIPT.slice(0, k).map((m) => {
    const en = typeof m.en === 'function' ? m.en() : m.en;
    const box = Object.assign(document.createElement('div'), { className: `msg msg--${m.who}` });
    box.append(Object.assign(document.createElement('div'), { className: 'msg__en', textContent: en }),
      Object.assign(document.createElement('div'), { className: 'msg__meta', textContent: m.who === 'you' ? m.meta : 'typed reply, signed back' }));
    return box;
  }));
  log.scrollTop = log.scrollHeight;
  const last = SCRIPT[k - 1];
  if (k > shown && last?.who === 'them') talkAvatar.play(last.tokens(), { onWord: (w) => { $('talk-caption').textContent = w; } }).then(() => { $('talk-caption').textContent = ''; });
  else if (k < shown || last?.who === 'you') talkAvatar.stop();
  shown = k;
}

// ---- one loop reads the engine's progress for both bespoke acts ----------------------------------------------
let lastLetter = -1;
(function tick() {
  const a = p('act-name');
  if (a > 0.04) {
    const i = Math.min(name.length - 1, Math.floor((a / 0.9) * name.length));
    if (mode === 'play' || i !== lastLetter) {
      mode = 'scrub';
      lastLetter = i;
      $('name-hint').textContent = 'Stepping through your name with the scroll wheel.';
      nameAvatar.show(name[i]);
      spelled(i);
    }
  } else if (mode === 'scrub') { lastLetter = -1; play(); }
  const k = AT.filter((t) => p('act-talk') >= t).length;
  if (k !== shown) renderTalk(k);
  requestAnimationFrame(tick);
}());

// ---- act 5: an actual input ------------------------------------------------------------------------------------
$('open-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const typed = $('open-name').value.trim();
  try {
    const res = await fetch('/api/profiles', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: typed }) });
    const profile = await res.json();
    if (res.ok) localStorage.setItem('sb.profile', JSON.stringify({ id: profile.id, name: profile.name }));
  } catch { /* the app asks for a name itself if this did not work */ }
  location.href = '/app';
});
