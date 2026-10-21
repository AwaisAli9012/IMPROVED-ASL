// Metrics: renders reports/*.json (written by scripts/evaluate.py). No number on this page is typed in by hand.
import { el, getJSON, initShell } from './api.js';

initShell();
const reports = await getJSON('/api/reports');
const pct = new Intl.NumberFormat('en', { style: 'percent', minimumFractionDigits: 1, maximumFractionDigits: 1 });
const ms = new Intl.NumberFormat('en', { maximumFractionDigits: 1 });
const dec = new Intl.NumberFormat('en', { maximumFractionDigits: 3 });
const TASKS = [['words', 'Words'], ['static', 'Letters and numbers'], ['emotion', 'Tone from the face']];

function testSentence(r) {
  const s = r.split;
  if (s.test_signers) return `${s.n_test} clips from ${s.test_signers.length} signers the model never saw, ${r.n_classes} words.`;
  return `${s.n_test} held-out test samples, ${r.n_classes} classes.`;
}

function kpis(r) {
  const f = (v, fmt = pct) => (v == null ? 'n/a' : fmt.format(v));
  const tiles = [['Accuracy', f(r.accuracy)], ['Top 3', f(r.top3)], ['Top 5', f(r.top5)], ['Macro F1', f(r.f1_macro)],
    ['Macro precision', f(r.precision_macro)], ['Macro recall', f(r.recall_macro)], ['Weighted F1', f(r.f1_weighted)],
    ['Balanced accuracy', f(r.balanced_accuracy)], ['Cohen kappa', f(r.cohen_kappa, dec)], ['MCC', f(r.mcc, dec)],
    ['Calibration error', f(r.ece_15bins)], ['Speed, median', `${ms.format(r.latency_ms_p50)} ms`], ['Speed, 95th', `${ms.format(r.latency_ms_p95)} ms`]];
  if (r.accept_threshold != null) tiles.push(['When it answers', f(r.precision_at_threshold)], ['…on this share', f(r.coverage_at_threshold)]);
  return el('div', { class: 'kpis' }, tiles.filter(([, v]) => v !== 'n/a').map(([label, value]) => el('div', { class: 'kpi' }, el('b', {}, value), el('span', { class: 'label' }, label))));
}

function table(headers, rows) {
  let sortBy = null, desc = true;
  const body = el('tbody');
  const paint = () => {
    const sorted = sortBy == null ? rows : [...rows].sort((a, b) => (typeof a[sortBy] === 'string' ? a[sortBy].localeCompare(b[sortBy]) : a[sortBy] - b[sortBy]) * (desc ? -1 : 1));
    body.replaceChildren(...sorted.map((row) => el('tr', {}, row.map((c, i) => el('td', {}, typeof c === 'number' && i < row.length - 1 ? pct.format(c) : String(c))))));
  };
  const head = el('tr', {}, headers.map((h, i) => el('th', { scope: 'col', tabindex: 0, 'aria-sort': 'none', onclick: (e) => {
    desc = sortBy === i ? !desc : true;
    sortBy = i;
    head.querySelectorAll('th').forEach((t) => t.setAttribute('aria-sort', 'none'));
    e.currentTarget.setAttribute('aria-sort', desc ? 'descending' : 'ascending');
    paint();
  }, onkeydown: (e) => e.key === 'Enter' && e.currentTarget.click() }, h)));
  paint();
  return el('div', { class: 'table-wrap' }, el('table', { class: 'table num' }, el('thead', {}, head), body));
}

function section(key, title) {
  const r = reports[key];
  if (!r) return el('div', { class: 'panel' }, el('h2', {}, title), el('p', { class: 'muted' }, 'Not trained yet. Run the training and evaluation scripts to fill this in.'));
  const classRows = Object.entries(r.per_class).map(([name, m]) => [name, m.precision, m.recall, m.f1, m.support]);
  return el('div', { class: 'panel' },
    el('h2', {}, title),
    el('p', { class: 'muted' }, testSentence(r)),
    kpis(r),
    el('p', { class: 'muted' }, r.notes),
    r.per_source ? table(['Source', 'Accuracy', 'Macro F1', 'Samples'], Object.entries(r.per_source).map(([s, m]) => [s, m.accuracy, m.f1_macro, m.n])) : null,
    el('div', { class: 'charts' },
      el('figure', { style: 'margin:0' }, el('img', { src: `/reports/${key}_confusion.png`, alt: `${title}: confusion matrix on the test set`, loading: 'lazy' }), el('figcaption', { class: 'label' }, 'Confusion matrix')),
      el('figure', { style: 'margin:0' }, el('img', { src: `/reports/${key}_per_class_f1.png`, alt: `${title}: F1 score for each class`, loading: 'lazy' }), el('figcaption', { class: 'label' }, 'F1 per class'))),
    table(['Class', 'Precision', 'Recall', 'F1', 'Support'], classRows),
    r.worst_confusions?.length ? el('p', { class: 'muted' }, 'Most confused: ' + r.worst_confusions.slice(0, 5).map(([t, p, n]) => `${t} read as ${p} (${n})`).join(', ') + '.') : null);
}

document.getElementById('reports').replaceChildren(...TASKS.map(([k, t]) => section(k, t)));
