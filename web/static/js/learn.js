// Course map: units as a path, with progress, streak and a review session when items are due.
import { el, fill, getJSON, icon, initShell, refreshIcons } from './api.js';

const profile = await initShell({ requireProfile: true });
const data = await getJSON(`/api/learn/${profile.id}`);
const { units } = data;
const total = units.reduce((n, u) => n + u.items.length, 0);
const mastered = units.reduce((n, u) => n + u.mastered, 0);
const passed = units.filter((u) => u.state === 'passed').length;

document.getElementById('hello').textContent = `Hello, ${data.profile.name}`;
document.getElementById('stats').replaceChildren(...[['XP', data.profile.xp], ['Day streak', data.profile.streak], ['Signs mastered', `${mastered}/${total}`]]
  .map(([label, value]) => el('div', { class: 'stat' }, el('b', {}, String(value)), el('span', { class: 'label' }, label))));
const bar = document.getElementById('course-bar');
bar.style.setProperty('--p', passed / units.length);
bar.setAttribute('aria-valuenow', Math.round((100 * passed) / units.length));

fill(document.getElementById('actions'),
  data.due ? el('a', { class: 'btn btn--primary', href: '/learn/review' }, icon('rotate-ccw'), `Review ${data.due} due`) : null,
  data.complete ? el('a', { class: 'btn btn--primary', href: '/certificate' }, icon('award'), 'Get your certificate') : null);

const ICON = { locked: 'lock', open: 'play', passed: 'check' };
const ACTION = { locked: 'Locked', open: 'Start', passed: 'Practise again' };
document.getElementById('path').replaceChildren(...units.map((u) => {
  const meta = u.items.length ? `${u.mastered} of ${u.items.length} mastered` : u.kind === 'intro' ? 'Five short pages' : 'Sign whole sentences';
  const action = u.state === 'locked'
    ? el('button', { class: 'btn btn--sm', type: 'button', disabled: true }, ACTION.locked)
    : el('a', { class: `btn btn--sm${u.state === 'open' ? ' btn--primary' : ''}`, href: `/learn/${u.id}` }, u.state === 'open' && u.mastered ? 'Continue' : ACTION[u.state]);
  return el('li', { class: 'unit', 'data-state': u.state },
    el('span', { class: 'unit__icon' }, icon(ICON[u.state])),
    el('div', {}, el('div', { class: 'unit__title' }, u.title), el('div', { class: 'muted' }, meta)),
    action);
}));
refreshIcons();
