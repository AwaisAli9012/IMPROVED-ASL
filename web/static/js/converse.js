// Converse page: you sign, SignBridge reads, builds an English sentence and speaks it; the other person
// types or speaks, and a signing figure answers.
import { announce, config, el, icon, initShell, postJSON, refreshIcons, toast } from './api.js';
import { Avatar } from './avatar.js';
import { needsOnboarding, runOnboarding } from './onboarding.js';
import { Recognizer, ToneTracker } from './recognizer.js';
import { Vision, wireCamera } from './vision.js';

initShell();
const $ = (id) => document.getElementById(id);
const WH = new Set(['what', 'where', 'who', 'why', 'how', 'when', 'which']);
const TONE = { neutral: ['meh', 'Neutral'], happy: ['smile', 'Happy'], sad: ['frown', 'Sad'], angry: ['angry', 'Angry'], surprised: ['sparkles', 'Surprised'] };
const VOICE = { neutral: [1, 1], happy: [1.08, 1.15], sad: [0.88, 0.9], angry: [1.05, 0.85], surprised: [1.1, 1.25] }; // [rate, pitch]
const QUESTION = { yesno: 'Question (yes/no)', wh: 'Question (what, where, who)' };

const cfg = await config();
const vision = new Vision($('video'), $('overlay'));
const avatar = new Avatar($('avatar'));
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
const recognizer = new Recognizer(cfg, {
  onGloss: addGloss,
  onSuggest: showSuggestions,
  onChar: (c) => announce(c),
  onBusy: (busy) => $('stage').setAttribute('aria-busy', busy),
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

// ---- speaking -----------------------------------------------------------------------------------
function speak(text, toneName = 'neutral') {
  if (!('speechSynthesis' in window)) return;
  const [rate, pitch] = VOICE[toneName] ?? VOICE.neutral;
  const u = Object.assign(new SpeechSynthesisUtterance(text), { lang: 'en-US', rate, pitch });
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

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
  const words = glosses.map((g) => g.display.startsWith('fs:') ? g.display : g.type === 'spell' ? g.label : g.display);
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

const Recog = window.SpeechRecognition ?? window.webkitSpeechRecognition;
if (!Recog) {
  $('mic').disabled = true;
  $('mic').title = 'Speech input needs Chrome or Edge';
} else {
  let rec = null;
  $('mic').addEventListener('click', () => {
    if (rec) return rec.stop();
    rec = Object.assign(new Recog(), { lang: 'en-US', interimResults: true });
    rec.onstart = () => $('mic').setAttribute('aria-pressed', 'true');
    rec.onresult = (e) => {
      const text = [...e.results].map((r) => r[0].transcript).join(' ');
      $('reply').value = text;
      if (e.results[e.results.length - 1].isFinal) { $('reply').value = ''; reply(text); }
    };
    rec.onerror = (e) => e.error !== 'no-speech' && e.error !== 'aborted' && toast(`Microphone: ${e.error}`, 'error');
    rec.onend = () => { rec = null; $('mic').setAttribute('aria-pressed', 'false'); };
    rec.start();
  });
}

$('speed').addEventListener('click', (e) => {
  if (!e.target.dataset.speed) return;
  avatar.setSpeed(+e.target.dataset.speed);
  document.querySelectorAll('#speed button').forEach((b) => b.setAttribute('aria-checked', b === e.target));
});
$('mirror').addEventListener('change', (e) => avatar.setMirror(e.target.checked));

$('export').addEventListener('click', () => {
  if (!transcript.length) return toast('Nothing to save yet.');
  const text = transcript.map((m) => `${m.who === 'you' ? 'Signer' : 'Reply'}: ${m.english}  (${m.detail})`).join('\n');
  const a = el('a', { href: URL.createObjectURL(new Blob([text], { type: 'text/plain' })), download: 'signbridge-conversation.txt' });
  a.click();
  URL.revokeObjectURL(a.href);
});

// ---- keyboard -------------------------------------------------------------------------------------
$('help').addEventListener('click', () => $('help-dialog').showModal());
document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, textarea, select') || e.metaKey || e.ctrlKey || e.altKey || document.querySelector('dialog[open]:not(#onboard)')) return;
  const k = e.key;
  const top = $('suggest')._top;
  if (k === 'w' || k === 'W') setMode('words');
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
