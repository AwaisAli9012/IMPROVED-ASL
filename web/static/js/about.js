// About: static explanation plus live numbers from /api/config, /api/curriculum, /api/reports and /api/health.
import { el, getJSON, initShell } from './api.js';

initShell();
const $ = (id) => document.getElementById(id);
const pct = new Intl.NumberFormat('en', { style: 'percent', minimumFractionDigits: 1, maximumFractionDigits: 1 });
const [cfg, units, reports, health] = await Promise.all([getJSON('/api/config', { quiet: true }), getJSON('/api/curriculum', { quiet: true }), getJSON('/api/reports', { quiet: true }), getJSON('/api/health', { quiet: true })].map((p) => p.catch(() => null)));
if (cfg) { $('a-words').textContent = cfg.words.length; $('a-digits').textContent = cfg.DIGITS.length; $('a-gemini').hidden = !cfg.gemini; }
if (units) $('a-units').textContent = units.length;
const rows = [['words', 'Words'], ['static', 'Letters and numbers'], ['emotion', 'Tone from the face']].filter(([k]) => reports?.[k]);
$('a-table').tBodies[0].replaceChildren(...rows.map(([k, name]) => {
  const r = reports[k], s = r.split;
  return el('tr', {}, el('td', {}, name), el('td', {}, s.test_signers ? `${s.n_test} clips, ${s.test_signers.length} unseen signers` : `${s.n_test} samples`),
    el('td', {}, pct.format(r.accuracy)), el('td', {}, pct.format(r.f1_macro)), el('td', {}, `${r.latency_ms_p95.toFixed(1)} ms`));
}));
if (health) $('a-status').textContent = `Status: ${health.status}. Models loaded: ${Object.entries(health.models).map(([k, v]) => `${k} ${v ? 'yes' : 'no'}`).join(', ')}.`;
