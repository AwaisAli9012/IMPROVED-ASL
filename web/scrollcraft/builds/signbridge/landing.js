// Landing behaviour. The signature move lives here: the visitor's name, fingerspelled by the real signing
// figure, stepped by the scroll wheel, and returned in the conversation. Every number comes from the API,
// and the points that assemble in act 2 are the real landmarks of a recorded sign.
import { Avatar } from '/static/js/avatar.js';
import { HAND_BONES } from '/static/js/vision.js';

const $ = (id) => document.getElementById(id);
const sign = (label) => ({ type: 'sign', label });
const get = (url) => fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(new Error(url)))).catch(() => null);
const p = (id) => parseFloat($(id).style.getPropertyValue('--sc-p')) || 0; // progress the engine publishes on each act
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ease = (t) => 1 - (1 - t) ** 3;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const css = getComputedStyle(document.documentElement);
const COLOR = { accent: css.getPropertyValue('--sc-accent').trim(), ink: css.getPropertyValue('--sc-ink').trim(), soft: css.getPropertyValue('--sc-ink-soft').trim() };

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

// ---- act 1 backdrop: a constellation of points that reacts to the pointer -----------------------------
function constellation(canvas) {
  const ctx = canvas.getContext('2d');
  const ratio = devicePixelRatio || 1;
  let W, H, dots, ptr = { x: -1e4, y: -1e4 }, visible = true;
  const fit = () => {
    W = canvas.width = canvas.clientWidth * ratio;
    H = canvas.height = canvas.clientHeight * ratio;
    dots = Array.from({ length: Math.round((W * H) / (ratio * ratio * 16000)) }, () => ({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - 0.5) * 0.3 * ratio, vy: (Math.random() - 0.5) * 0.3 * ratio }));
  };
  const draw = () => {
    ctx.clearRect(0, 0, W, H);
    const reach = 130 * ratio;
    for (const d of dots) {
      d.x = (d.x + d.vx + W) % W;
      d.y = (d.y + d.vy + H) % H;
      const dx = d.x - ptr.x, dy = d.y - ptr.y, near = Math.hypot(dx, dy) < 160 * ratio;
      if (near) { d.x += dx * 0.012; d.y += dy * 0.012; } // points are nudged away from the pointer
      ctx.fillStyle = near ? COLOR.accent : COLOR.soft;
      ctx.globalAlpha = near ? 0.9 : 0.35;
      ctx.beginPath(); ctx.arc(d.x, d.y, (near ? 2.6 : 1.8) * ratio, 0, 7); ctx.fill();
    }
    ctx.lineWidth = ratio;
    ctx.strokeStyle = COLOR.soft;
    for (let i = 0; i < dots.length; i++) for (let j = i + 1; j < dots.length; j++) {
      const dist = Math.hypot(dots[i].x - dots[j].x, dots[i].y - dots[j].y);
      if (dist < reach) { ctx.globalAlpha = (1 - dist / reach) * 0.22; ctx.beginPath(); ctx.moveTo(dots[i].x, dots[i].y); ctx.lineTo(dots[j].x, dots[j].y); ctx.stroke(); }
    }
  };
  const loop = () => { if (visible) draw(); if (!reduce) requestAnimationFrame(loop); };
  new ResizeObserver(() => { fit(); draw(); }).observe(canvas);
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(canvas);
  addEventListener('pointermove', (e) => { const b = canvas.getBoundingClientRect(); ptr = { x: (e.clientX - b.left) * ratio, y: (e.clientY - b.top) * ratio }; });
  fit();
  loop();
}
constellation($('field'));

// ---- act 1: your name, signed ------------------------------------------------------------------------
const clean = (s) => s.toUpperCase().replace(/[^A-Z]/g, '').slice(0, 14);
const title = (s) => s[0] + s.slice(1).toLowerCase();
let name = clean($('name-input').value) || 'ADA';
let mode = 'play';
const nameAvatar = new Avatar($('name-avatar'));

function tiles(current = -1) {
  const list = $('tiles');
  if (list.children.length !== name.length || list.dataset.name !== name) { // rebuild only when the name changes, so tiles keep their entrance animation
    list.dataset.name = name;
    list.replaceChildren(...[...name].map((c, i) => Object.assign(document.createElement('li'), { className: 'tile', textContent: c, style: `--i:${i}` })));
  }
  [...list.children].forEach((t, i) => t.setAttribute('aria-current', String(i === current)));
  $('spell-count').textContent = current >= 0 ? `${current + 1} of ${name.length}` : `${name.length} letters`;
}
function play() {
  mode = 'play';
  $('name-hint').textContent = 'Type a name. Scroll to step through its letters.';
  nameAvatar.play([{ type: 'spell', text: name }], { loop: true, onWord: (_, i) => tiles(i) });
}
$('name-form').addEventListener('submit', (e) => {
  e.preventDefault();
  name = clean($('name-input').value) || name;
  $('open-name').value = title(name);
  tiles();
  play();
});
$('name-input').addEventListener('input', () => { const n = clean($('name-input').value); if (n) { name = n; tiles(); play(); } });
tiles();
play();
$('open-name').value = title(name);

// ---- act 2: the real points of a recorded sign fly together, then play the sign -----------------------------
const pointsCanvas = $('points');
const demo = await get('/static/signs/hello.json');
if (!demo) $('points-box').hidden = true; // no animations exported yet: the page still reads without it
const seeded = (i) => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }; // fixed scatter, no flicker
function drawPoints(progress) {
  const ctx = pointsCanvas.getContext('2d');
  const r = devicePixelRatio || 1;
  const W = (pointsCanvas.width = pointsCanvas.clientWidth * r), H = (pointsCanvas.height = pointsCanvas.clientHeight * r);
  ctx.clearRect(0, 0, W, H);
  const n = demo.frames.length;
  const frame = demo.frames[Math.min(n - 1, Math.floor(clamp((progress - 0.12) / 0.8) * (n - 1)))];
  const gather = ease(clamp(progress / 0.3));
  const u = Math.min(W * 0.26, H * 0.22);
  const at = ([x, y]) => [W / 2 + x * u, H * 0.4 + y * u];
  const groups = [['pose', COLOR.ink, 3.4], ['lh', COLOR.soft, 2.6], ['rh', COLOR.accent, 2.6], ['lips', COLOR.accent, 1.8]];
  const place = {};
  let count = 0, seed = 0;
  for (const [key, color, size] of groups) {
    const pts = frame[key];
    if (!pts) continue;
    place[key] = pts.map((pt) => {
      const [fx, fy] = at(pt);
      const sx = seeded(seed++) * W, sy = seeded(seed++) * H; // each point starts at its own fixed scatter position
      count++;
      return [sx + (fx - sx) * gather, sy + (fy - sy) * gather];
    });
    ctx.fillStyle = color;
    for (const [x, y] of place[key]) { ctx.beginPath(); ctx.arc(x, y, size * r, 0, 7); ctx.fill(); }
  }
  const lines = clamp((gather - 0.75) / 0.25); // bones fade in once the points have arrived
  ctx.globalAlpha = lines * 0.8;
  ctx.lineWidth = 1.6 * r;
  ctx.lineCap = 'round';
  const seg = (pts, a, b, color) => { ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); ctx.stroke(); };
  for (const [key, color] of [['lh', COLOR.soft], ['rh', COLOR.accent]]) if (place[key]) for (const [a, b] of HAND_BONES) seg(place[key], a, b, color);
  if (place.pose) for (const [a, b] of [[1, 2], [1, 3], [3, 5], [2, 4], [4, 6]]) seg(place.pose, a, b, COLOR.ink);
  ctx.globalAlpha = 1;
  $('points-caption').textContent = gather < 1 ? `${count} points finding their place` : `${count} points, frame ${Math.min(n, Math.floor(clamp((progress - 0.12) / 0.8) * (n - 1)) + 1)} of ${n}`;
}

// ---- act 3: the conversation (scripted sample, real signing) --------------------------------------------
const talkAvatar = new Avatar($('talk-avatar'));
const SCRIPT = [
  { who: 'you', glosses: ['you', 'hungry'], grammar: 'Yes/no question', tone: 'Happy', en: 'Are you hungry?' },
  { who: 'them', en: 'Yes, I want to eat.', tokens: () => ['yes', 'me', 'want', 'eat'].map(sign) },
  { who: 'you', glosses: ['me', 'go', 'home'], grammar: '', tone: 'Neutral', en: 'I am going home.' },
  { who: 'them', en: 'See you tomorrow.', tokens: () => ['see', 'you', 'tomorrow'].map(sign) },
  { who: 'you', glosses: ['who', 'you'], grammar: 'What question', tone: 'Neutral', en: 'Who are you?' },
  { who: 'them', en: () => `I am ${title(name)}.`, tokens: () => [sign('me'), { type: 'spell', text: name }] },
];
const AT = [0.16, 0.30, 0.44, 0.58, 0.72, 0.85]; // the first 16% is a held surface, by design
let shown = 0;
const node = (tag, className, text) => Object.assign(document.createElement(tag), { className, textContent: text ?? '' });

function message(m, fresh) {
  const box = node('div', `msg msg--${m.who}${fresh ? ' is-new' : ''}`);
  if (m.who === 'you') {
    box.append(Object.assign(node('div', 'msg__glosses'), {}));
    m.glosses.forEach((g, i) => { const t = node('span', 'tag tag--gloss', g); t.style.setProperty('--i', i); box.lastChild.append(t); });
    const tags = node('div', 'msg__tags');
    if (m.grammar) tags.append(node('span', 'tag tag--accent', m.grammar));
    tags.append(node('span', 'tag', m.tone));
    box.append(tags);
  }
  const wave = node('span', 'wave');
  wave.append(...[0, 1, 2, 3, 4].map((i) => { const b = document.createElement('i'); b.style.setProperty('--i', i); return b; }));
  const line = node('div', 'msg__en', typeof m.en === 'function' ? m.en() : m.en);
  box.append(line);
  box.append(Object.assign(node('div', 'msg__meta'), {}));
  box.lastChild.append(m.who === 'you' ? wave : document.createTextNode('typed reply, signed back'));
  if (m.who === 'you') box.lastChild.append(document.createTextNode(' spoken aloud'));
  return box;
}

function renderTalk(k) {
  const log = $('talk-log');
  log.replaceChildren(...SCRIPT.slice(0, k).map((m, i) => message(m, i === k - 1 && k > shown)));
  log.scrollTop = log.scrollHeight;
  $('talk-step').textContent = k ? `Scripted sample · ${k} of ${SCRIPT.length}` : 'Scripted sample. The app does this live.';
  const last = SCRIPT[k - 1];
  const frame = $('talk-frame');
  if (k > shown && last?.who === 'them') {
    frame.classList.add('is-signing');
    talkAvatar.play(last.tokens(), { onWord: (w) => { $('talk-caption').textContent = w; } }).then(() => { $('talk-caption').textContent = ''; frame.classList.remove('is-signing'); });
  } else if (k < shown || last?.who === 'you') { talkAvatar.stop(); frame.classList.remove('is-signing'); }
  shown = k;
}

// ---- one loop reads the engine's progress for every bespoke act -----------------------------------------------
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
      tiles(i);
    }
  } else if (mode === 'scrub') { lastLetter = -1; play(); }

  const r = p('act-reads');
  if (demo) drawPoints(r);
  const step = r < 0.27 ? 0 : r < 0.51 ? 1 : r < 0.75 ? 2 : 3;
  [...$('rail').children].forEach((li, i) => { li.dataset.on = String(i <= step); li.dataset.now = String(i === step); });

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
