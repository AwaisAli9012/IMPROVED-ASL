// Dictionary: every supported sign, searchable and filterable, played by the signing figure.
import { el, getJSON, initShell, refreshIcons } from './api.js';
import { Avatar } from './avatar.js';

initShell();
const [units, cfg] = await Promise.all([getJSON('/api/curriculum'), getJSON('/api/config')]);
const displayOf = Object.fromEntries(cfg.words.map((w) => [w.label, w.display]));
const entries = units.flatMap((u) => (u.kind === 'hand' || u.kind === 'word' ? u.items.map((item) => ({
  label: item, display: displayOf[item] ?? item, unit: u.id, topic: u.kind === 'hand' ? (/[0-9]/.test(item) ? 'Numbers' : 'Alphabet') : u.title, hand: u.kind === 'hand' })) : []));
const topics = ['All', ...new Set(entries.map((e) => e.topic))];

const avatar = new Avatar(document.getElementById('avatar'));
let topic = 'All', selected = null;

function renderList() {
  const q = document.getElementById('search').value.trim().toLowerCase();
  const shown = entries.filter((e) => (topic === 'All' || e.topic === topic) && (!q || e.display.toLowerCase().includes(q) || e.label.includes(q)));
  document.getElementById('count').textContent = `${shown.length} of ${entries.length} signs`;
  document.getElementById('list').replaceChildren(...shown.map((e) => el('li', {}, el('button', {
    class: 'dict__item', type: 'button', 'aria-selected': e.label === selected?.label, onclick: () => select(e) },
  el('span', {}, e.display), el('span', { class: 'label' }, e.topic)))));
  if (!shown.length) document.getElementById('list').append(el('li', { class: 'muted', style: 'padding:var(--sc-4)' }, 'No sign matches.'));
}

function select(e) {
  selected = e;
  history.replaceState(null, '', `#${encodeURIComponent(e.label)}`);
  document.getElementById('name').textContent = e.display;
  const link = document.getElementById('practise');
  link.href = `/learn/${e.unit}`;
  link.hidden = false;
  avatar.play([{ type: 'sign', label: e.label }], { loop: true });
  renderList();
}

document.getElementById('filters').replaceChildren(...topics.map((t) => el('button', {
  class: 'chip chip--plain', type: 'button', 'aria-pressed': t === topic, onclick: (ev) => {
    topic = t;
    document.querySelectorAll('#filters button').forEach((b) => b.setAttribute('aria-pressed', b === ev.currentTarget));
    renderList();
  } }, t)));
document.getElementById('search').addEventListener('input', renderList);
document.getElementById('speed').addEventListener('click', (e) => {
  if (!e.target.dataset.speed) return;
  avatar.setSpeed(+e.target.dataset.speed);
  document.querySelectorAll('#speed button').forEach((b) => b.setAttribute('aria-checked', b === e.target));
  if (selected) select(selected);
});
document.getElementById('mirror').addEventListener('change', (e) => avatar.setMirror(e.target.checked));

renderList();
const hash = decodeURIComponent(location.hash.slice(1));
const start = entries.find((e) => e.label === hash);
if (start) select(start);
refreshIcons();
