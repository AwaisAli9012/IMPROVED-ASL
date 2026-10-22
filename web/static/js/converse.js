// Converse page: you sign, SignBridge reads, builds an English sentence and speaks it; the other person
// types or speaks, and a signing figure answers.
import { announce, config, download, el, fill, icon, initShell, listen, postJSON, refreshIcons, speak, toast } from './api.js';
import { Avatar, syncAvatarControls } from './avatar.js';
import { needsOnboarding, runOnboarding } from './onboarding.js';
import { Recognizer, ToneTracker } from './recognizer.js';
import { Vision, wireCamera } from './vision.js';

initShell();
const $ = (id) => document.getElementById(id);
const WH = new Set(['what', 'where', 'who', 'why', 'how', 'when', 'which']);
const TONE = { neutral: ['meh', 'Neutral'], happy: ['smile', 'Happy'], sad: ['frown', 'Sad'], angry: ['angry', 'Angry'], surprised: ['sparkles', 'Surprised'] };
const QUESTION = { yesno: 'Question (yes/no)', wh: 'Question (what, where, who)' };

const cfg = await config();
const vision = new Vision($('video'), $('overlay'));
const avatar = new Avatar($('avatar'));
syncAvatarControls(avatar);
const glosses = [];       // {type, label, display, question}
const transcript = [];    // for the export
let suggestTimer;

// ---- building the sentence ---------------------------------------------------------------------
const utteranceQuestion = () =>
  glosses.some((g) => g.type === 'sign' && WH.has(g.label)) ? 'wh' : glosses.find((g) => g.question !== 'none')?.question ?? 'none';

function renderBuilding() {
  $('building').replaceChildren(...glosses.map((g, i) => el('span', { class: `chip${g.type === 'spell' ? ' chip--spell' : ''}` }, g.display,
    el('button', { type: 'button', 'aria-label': `Remove ${g.display}`, onclick: () => { glosses.splice(i, 1); renderBuilding(); } }, icon('x')))));
  if (!glosses.length) $('building').append(el('span', { class: 'muted' }, 'Signs you make appear here.'));
  const q = utteranceQuestion();
  $('grammar').textContent = QUESTION[q] ?? '';
  for (const id of ['speak', 'undo', 'clear']) $(id).disabled = !glosses.length;
  refreshIcons();
}

function addGloss(g) {
  glosses.push(g);
  clearSuggestions();
  renderBuilding();
  announce(`Added ${g.display}`);
}

function clearSuggestions() {
  clearTimeout(suggestTimer);
  $('suggest').hidden = true;
  $('suggest').replaceChildren();
}

function showSuggestions(top, question) {
  clearSuggestions();
  $('suggest').replaceChildren(el('span', { class: 'label' }, 'Did you mean'), ...top.map((t, i) =>
    el('button', { class: 'chip chip--plain', type: 'button', onclick: () => addGloss({ type: 'sign', label: t.label, display: t.display, question }) }, `${i + 1}  ${t.display}`)));
  $('suggest').hidden = false;
  suggestTimer = setTimeout(clearSuggestions, 6000);
  $('suggest')._top = top.map((t) => ({ type: 'sign', label: t.label, display: t.display, question }));
}

// ---- recognition --------------------------------------------------------------------------------
let last = null; // the latest prediction, shown by the inspector
const recognizer = new Recognizer(cfg, {
  onGloss: addGloss,
  onSuggest: showSuggestions,
  onChar: (c) => announce(c),
  onBusy: (busy) => $('stage').setAttribute('aria-busy', busy),
  onResult: (r) => { last = r; },
  onError: (e) => { if (e.status === 503) toast('The sign model is not trained yet.', 'error'); },
  onCandidate: (c) => {
    $('cand-word').textContent = c?.display ?? '';
    $('cand-meter').dataset.state = c?.accept ? 'ok' : '';
    $('cand-meter').style.setProperty('--p', c?.prob ?? 0);
  },
});
const tone = new ToneTracker(cfg, (name, ema) => {
  const [ic, label] = TONE[name] ?? TONE.neutral;
  $('tone-icon').setAttribute('data-lucide', ic);
  $('tone-label').textContent = `${label} ${Math.round((ema[name] ?? 0) * 100)}%`;
  refreshIcons();
});
vision.on((ev) => { recognizer.push(ev); tone.push(ev.blend); });

function setMode(mode) {
  recognizer.setMode(mode);
  document.querySelectorAll('#mode button').forEach((b) => b.setAttribute('aria-checked', b.dataset.mode === mode));
  announce({ words: 'Words mode', letters: 'Letters mode', digits: 'Numbers mode' }[mode]);
}
$('mode').addEventListener('click', (e) => e.target.dataset.mode && setMode(e.target.dataset.mode));

// ---- model inspector ---------------------------------------------------------------------------
function paintInspector() {
  if ($('inspector').hidden) return;
  $('i-fps').textContent = `${vision.fps.toFixed(0)} fps`;
  if (!last) return;
  $('i-ms').textContent = `${last.ms.toFixed(0)} ms`;
  $('i-frames').textContent = last.frames ? `${last.frames}` : 'one frame';
  fill($('i-top'), ...last.top.map((t) => el('div', { class: 'inspector__bar' }, el('span', {}, t.display),
    el('i', { class: 'meter', style: `--p:${t.prob}` }, el('i')), el('b', { class: 'num' }, `${Math.round(t.prob * 100)}%`))));
}
function toggleInspector() {
  $('inspector').hidden = !$('inspector').hidden;
  $('inspect').setAttribute('aria-pressed', String(!$('inspector').hidden));
  paintInspector();
}
$('inspect').addEventListener('click', toggleInspector);
setInterval(paintInspector, 400);

// ---- speaking -----------------------------------------------------------------------------------
function addMessage(who, english, detail, toneName, tokens) {
  transcript.push({ who, english, detail });
  const meta = el('div', { class: 'msg__meta' }, detail,
    who === 'you'
      ? el('button', { class: 'btn btn--ghost btn--icon btn--sm', type: 'button', 'aria-label': 'Replay voice', onclick: () => speak(english, toneName) }, icon('volume-2'))
      : el('button', { class: 'btn btn--ghost btn--icon btn--sm', type: 'button', 'aria-label': 'Replay signs', onclick: () => playSigns(tokens) }, icon('play')));
  const box = $('transcript');
  box.append(el('div', { class: `msg msg--${who === 'you' ? 'you' : 'them'}` }, el('div', { class: 'msg__en' }, english), meta));
  box.scrollTop = box.scrollHeight;
  refreshIcons();
}

async function speakSentence() {
  if (!glosses.length) return;
  const toneName = tone.takeUtteranceTone();
  const words = glosses.map((g) => (g.type === 'spell' ? g.label : g.display));
  try {
    const { english } = await postJSON('/api/sentence', { glosses: words, tone: toneName, question: utteranceQuestion() });
    addMessage('you', english, `${glosses.map((g) => g.display).join(' ')} · ${TONE[toneName]?.[1] ?? 'Neutral'}`, toneName);
    speak(english, toneName);
    announce(english);
    glosses.length = 0;
    renderBuilding();
  } catch { /* api.js already showed the error */ }
}
$('speak').addEventListener('click', speakSentence);
$('undo').addEventListener('click', () => { glosses.pop(); renderBuilding(); });
$('clear').addEventListener('click', () => { glosses.length = 0; renderBuilding(); });

// ---- the other person: typing or speaking, answered with signing ---------------------------------
const playSigns = (tokens) => avatar.play(tokens, { onWord: (w) => { $('caption').textContent = w; } }).then(() => { $('caption').textContent = ''; });

async function reply(text) {
  text = text.trim();
  if (!text) return;
  try {
    const { tokens } = await postJSON('/api/gloss', { text });
    const detail = tokens.map((t) => (t.type === 'sign' ? t.label : t.text.split('').join('-'))).join(' ');
    addMessage('them', text, detail, 'neutral', tokens);
    playSigns(tokens);
  } catch { /* api.js already showed the error */ }
}
$('composer').addEventListener('submit', (e) => { e.preventDefault(); const t = $('reply').value; $('reply').value = ''; reply(t); });

let rec = null;
if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
  $('mic').disabled = true;
  $('mic').title = 'Speech input needs Chrome or Edge';
}
$('mic').addEventListener('click', () => {
  if (rec) return rec.stop();
  rec = listen({
    onState: (on) => { $('mic').setAttribute('aria-pressed', on); if (!on) rec = null; },
    onResult: ({ finals, interim }) => { $('reply').value = interim; finals.forEach((f) => { $('reply').value = ''; reply(f); }); },
  });
});

$('speed').addEventListener('click', (e) => {
  if (!e.target.dataset.speed) return;
  avatar.setSpeed(+e.target.dataset.speed);
  document.querySelectorAll('#speed button').forEach((b) => b.setAttribute('aria-checked', b === e.target));
});
$('mirror').addEventListener('change', (e) => avatar.setMirror(e.target.checked));

$('export').addEventListener('click', () => {
  if (!transcript.length) return toast('Nothing to save yet.');
  download('signbridge-conversation.txt', transcript.map((m) => `${m.who === 'you' ? 'Signer' : 'Reply'}: ${m.english}  (${m.detail})`).join('\n'));
});

// ---- keyboard -------------------------------------------------------------------------------------
$('help').addEventListener('click', () => $('help-dialog').showModal());
document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, textarea, select') || e.metaKey || e.ctrlKey || e.altKey || document.querySelector('dialog[open]:not(#onboard)')) return;
  const k = e.key;
  const top = $('suggest')._top;
  if (k === 'i' || k === 'I') toggleInspector();
  else if (k === 'w' || k === 'W') setMode('words');
  else if (k === 'a' || k === 'A') setMode('letters');
  else if (k === 'n' || k === 'N') setMode('digits');
  else if (k === ' ') { e.preventDefault(); recognizer.endSegment(); }
  else if (k === 'Enter') { e.preventDefault(); speakSentence(); }
  else if (k === 'Backspace') { glosses.pop(); renderBuilding(); }
  else if (k === 'Escape') { $('suggest').hidden ? (glosses.length = 0, renderBuilding()) : clearSuggestions(); }
  else if ('123'.includes(k) && !$('suggest').hidden && top?.[+k - 1]) addGloss(top[+k - 1]);
});

// ---- camera ----------------------------------------------------------------------------------------
wireCamera(vision, { onReady: () => { if (needsOnboarding()) runOnboarding(vision); } });
$('recalibrate').addEventListener('click', () => (vision.running ? runOnboarding(vision, { neutralOnly: true }) : toast('Start the camera first.')));
renderBuilding();
window.sb = { vision, recognizer }; // handy in the console: sb.vision.fps
