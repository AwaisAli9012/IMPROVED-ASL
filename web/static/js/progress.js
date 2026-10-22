// Progress: mastery heatmap, last 14 days, weak spots and badges, all from /api/progress/<id>.
import { el, fill, getJSON, icon, initShell, refreshIcons } from './api.js';

const profile = await initShell({ requireProfile: true });
const r = await getJSON(`/api/progress/${profile.id}`);
const pct = new Intl.NumberFormat('en', { style: 'percent', maximumFractionDigits: 0 });
const t = r.totals;

document.getElementById('sub').textContent = `${r.profile.name}: ${t.seen} of ${t.signs} signs tried, ${t.mastered} mastered.`;
fill(document.getElementById('stats'), ...[['XP', r.profile.xp], ['Best streak', `${r.profile.longest_streak} d`], ['Accuracy', t.accuracy == null ? 'n/a' : pct.format(t.accuracy)]]
  .map(([label, value]) => el('div', { class: 'stat' }, el('b', {}, String(value)), el('span', { class: 'label' }, label))));

const peak = Math.max(1, ...r.activity.map((d) => d.n));
fill(document.getElementById('activity'), ...r.activity.map((d) => el('div', { class: 'activity__col', title: `${d.day}: ${d.n} signs` },
  el('i', { style: `--h:${d.n / peak}` }), el('span', { class: 'label' }, d.day.slice(8)))));
document.getElementById('activity-note').textContent = r.activity.every((d) => !d.n) ? 'Practise a sign in Learn and it shows up here.' : `${r.activity.reduce((n, d) => n + d.n, 0)} signs practised in two weeks.`;

fill(document.getElementById('weak'), r.weak.length
  ? r.weak.map((w) => el('a', { class: 'row row--between', href: `/dictionary#${encodeURIComponent(w.item)}` }, el('b', {}, w.display), el('span', { class: 'muted num' }, `${w.correct} of ${w.attempts} right`)))
  : el('p', { class: 'muted' }, 'Nothing yet. Signs you miss twice or more appear here so you can revisit them.'));

const BADGE_ICON = { first_sign: 'sparkles', ten_signs: 'hand', fifty_correct: 'star', ten_mastered: 'medal', alphabet: 'type', numbers: 'hash', streak3: 'flame', streak7: 'flame', perfect: 'target', course: 'graduation-cap' };
fill(document.getElementById('badges'), ...r.badges.map((b) => el('li', { class: 'badge', 'data-earned': b.earned, title: b.detail },
  icon(BADGE_ICON[b.id] ?? 'award'), el('b', {}, b.title), el('span', { class: 'muted' }, b.earned ? 'Earned' : b.detail))));

fill(document.getElementById('heat'), ...r.units.map((u) => el('div', { class: 'stack' }, el('h3', {}, u.title),
  el('div', { class: 'cells' }, u.items.map((i) => el('a', { class: 'cell cell--link', href: `/dictionary#${encodeURIComponent(i.item)}`, style: `--b:${i.box}`,
    title: `${i.display}: ${i.attempts ? `${i.correct} of ${i.attempts} right` : 'not tried yet'}`, 'aria-label': `${i.display}, level ${i.box} of 5` }, i.display.slice(0, 3)))))));
refreshIcons();
