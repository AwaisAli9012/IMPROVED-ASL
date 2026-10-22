// Live captions: continuous speech to large text, optionally signed sentence by sentence.
import { download, el, fill, getPrefs, initShell, listen, postJSON, refreshIcons, setPref, toast } from './api.js';
import { Avatar, syncAvatarControls } from './avatar.js';

initShell();
const $ = (id) => document.getElementById(id);
const log = [];
let rec = null, wanted = false;
const size = () => document.documentElement.style.setProperty('--cap', `${1.1 + getPrefs().captionSize * 0.55}rem`);
size();

if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) { $('unsupported').hidden = false; $('mic').disabled = true; }
$('smaller').addEventListener('click', () => { setPref('captionSize', Math.max(0, getPrefs().captionSize - 1)); size(); });
$('larger').addEventListener('click', () => { setPref('captionSize', Math.min(8, getPrefs().captionSize + 1)); size(); });

let avatar = null;
$('sign').addEventListener('change', (e) => {
  $('sign-box').hidden = !e.target.checked;
  if (e.target.checked && !avatar) { avatar = new Avatar($('avatar')); syncAvatarControls(avatar); }
});

function paint(interim = '') {
  fill($('captions'), ...log.map((l) => el('p', { class: 'captions__line' }, el('span', { class: 'label num' }, l.at), ' ', l.text)),
    interim ? el('p', { class: 'captions__line captions__line--live' }, interim) : null);
  window.scrollTo({ top: document.body.scrollHeight });
}

function start() {
  wanted = true;
  rec = listen({
    continuous: true,
    onState: (on) => {
      $('mic').setAttribute('aria-pressed', on);
      $('mic-label').textContent = on ? 'Stop listening' : 'Start listening';
      if (!on && wanted) setTimeout(start, 300); // the browser ends a session after a pause: keep going until Stop
    },
    onResult: async ({ finals, interim }) => {
      for (const text of finals) {
        log.push({ text, at: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) });
        if ($('sign').checked && avatar) postJSON('/api/gloss', { text }, { quiet: true }).then(({ tokens }) => avatar.play(tokens, { onWord: (w) => { $('caption').textContent = w; } })).catch(() => {});
      }
      paint(interim);
    },
  });
}
$('mic').addEventListener('click', () => {
  if (wanted) { wanted = false; rec?.stop(); return; }
  start();
});
$('clear').addEventListener('click', () => { log.length = 0; paint(); });
$('save').addEventListener('click', () => (log.length ? download('captions.txt', log.map((l) => `${l.at}  ${l.text}`).join('\n')) : toast('Nothing to save yet.')));
paint();
refreshIcons();
