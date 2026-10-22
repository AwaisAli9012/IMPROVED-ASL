// Phrasebook: curated and personal phrases. Each one is signed by the figure and (optionally) spoken.
import { el, fill, getJSON, icon, initShell, postJSON, refreshIcons, speak } from './api.js';
import { Avatar, syncAvatarControls } from './avatar.js';

initShell();
const $ = (id) => document.getElementById(id);
const avatar = new Avatar($('avatar'));
syncAvatarControls(avatar);
const MINE = 'sb.phrases';
const mine = () => { try { return JSON.parse(localStorage.getItem(MINE)) ?? []; } catch { return []; } };
const saveMine = (list) => { try { localStorage.setItem(MINE, JSON.stringify(list)); } catch { /* private mode: kept for this page only */ } };
const groups = await getJSON('/api/phrases');
let playing = 0;

async function say(text) {
  const id = ++playing;
  if ($('aloud').checked) speak(text);
  const { tokens } = await postJSON('/api/gloss', { text });
  if (id === playing) avatar.play(tokens, { onWord: (w) => { $('caption').textContent = w; } }).then(() => { if (id === playing) $('caption').textContent = ''; });
}

function render() {
  const q = $('search').value.trim().toLowerCase();
  const all = [...groups, { title: 'My phrases', items: mine(), own: true }];
  fill($('groups'), ...all.map((g) => {
    const shown = g.items.filter((t) => !q || t.toLowerCase().includes(q));
    return shown.length || (g.own && !q) ? el('div', { class: 'stack' }, el('h2', {}, g.title),
      el('div', { class: 'choice' }, shown.map((t) => el('span', { class: 'phrase' },
        el('button', { class: 'chip chip--plain', type: 'button', onclick: () => say(t) }, t),
        g.own ? el('button', { class: 'phrase__x', type: 'button', 'aria-label': `Remove ${t}`, onclick: () => { saveMine(mine().filter((x) => x !== t)); render(); } }, icon('x')) : null)),
      g.own && !shown.length ? el('p', { class: 'muted' }, 'Phrases you add appear here.') : null)) : null;
  }));
  refreshIcons();
}
$('search').addEventListener('input', render);
$('add').addEventListener('submit', (e) => {
  e.preventDefault();
  const text = $('own').value.trim();
  if (text && !mine().includes(text)) { saveMine([...mine(), text]); $('own').value = ''; render(); say(text); }
});
$('speed').addEventListener('click', (e) => {
  if (!e.target.dataset.speed) return;
  avatar.setSpeed(+e.target.dataset.speed);
  document.querySelectorAll('#speed button').forEach((b) => b.setAttribute('aria-checked', b === e.target));
});
$('mirror').addEventListener('change', (e) => avatar.setMirror(e.target.checked));
render();
