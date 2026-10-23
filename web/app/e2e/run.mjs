// Browser checks against a running server: node e2e/run.mjs   (E2E_BASE defaults to http://127.0.0.1:5000)
// Uses the system Chrome and a fake camera. Exits non-zero on any failure.
import { readFileSync } from 'node:fs';
import { chromium } from 'playwright-core';

const BASE = process.env.E2E_BASE ?? 'http://127.0.0.1:5000';
const CHROME = process.env.CHROME ?? '/home/worker-male/.local/bin/google-chrome';
const PAGES = ['/', '/app', '/learn', '/learn/start', '/learn/abc1', '/learn/review', '/progress', '/phrasebook', '/captions', '/dictionary', '/metrics', '/settings', '/about', '/certificate'];
const AXE = readFileSync(new URL('../node_modules/axe-core/axe.min.js', import.meta.url), 'utf8');
const failures = [];
const ok = (cond, msg) => { if (!cond) { failures.push(msg); console.log('  FAIL', msg); } };
const section = (t) => console.log(`\n${t}`);

const browser = await chromium.launch({ executablePath: CHROME, args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--no-sandbox'] });
const context = async (opts = {}) => {
  const c = await browser.newContext({ viewport: { width: 1280, height: 800 }, permissions: ['camera'], ...opts });
  return c;
};
const profile = async (page) => { // create a learner through the dialog, then wait for the save (the next navigation would race it)
  await page.goto(`${BASE}/learn`);
  await page.waitForSelector('dialog[open] input[name], dialog[open] input');
  await page.fill('dialog[open] input', 'Tester');
  await page.click('dialog[open] button[type=submit]');
  await page.waitForFunction(() => localStorage.getItem('sb.profile'));
};

// ---- 1. every page loads cleanly --------------------------------------------------------------------
section('pages');
{
  const c = await context(), page = await c.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(`pageerror ${e.message}`));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource|^INFO:/.test(m.text()) && errors.push(`console ${m.text()}`));
  page.on('requestfailed', (r) => errors.push(`failed ${r.url()}`));
  page.on('response', (r) => r.status() >= 400 && !/\/api\/learn\/\d+\/review|favicon/.test(r.url()) && errors.push(`${r.status()} ${r.url()}`));
  await profile(page);
  for (const path of PAGES) {
    errors.length = 0;
    await page.goto(BASE + path);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(400);
    ok(!errors.length, `${path}: ${errors.join(' | ')}`);
    console.log(' ', path, errors.length ? 'errors' : 'ok');
  }
  await c.close();
}

// ---- 2. flows -------------------------------------------------------------------------------------
section('flows');
{
  const c = await context(), page = await c.newPage();
  await profile(page);
  await page.goto(`${BASE}/learn/start`);
  for (let i = 0; i < 5; i++) { await page.waitForSelector('h2'); await page.click('button.btn--primary:has-text("Next"), button.btn--primary:has-text("Finish")'); }
  ok(await page.waitForSelector('text=Unit complete', { timeout: 8000 }).then(() => true).catch(() => false), 'intro unit completes');
  await page.goto(`${BASE}/learn/abc1`);
  ok(await page.waitForSelector("button:has-text(\"I'm ready\")", { timeout: 15000 }).then(() => true).catch(() => false), 'abc1 shows the Watch step');
  await page.goto(`${BASE}/learn/greetings`);
  await page.waitForURL(/\/learn$/, { timeout: 6000 }).catch(() => {});
  ok(/\/learn$/.test(page.url()), 'a locked unit redirects to /learn');
  await page.goto(`${BASE}/certificate`);
  ok(await page.waitForSelector('text=Not yet').then(() => true).catch(() => false), 'certificate is locked');
  await page.goto(`${BASE}/dictionary#hello`);
  ok(await page.waitForSelector('.dict__item[aria-current="true"]:has-text("hello")', { timeout: 6000 }).then(() => true).catch(() => false), 'dictionary #hello selects hello');
  await page.goto(`${BASE}/app`);
  await page.fill('input[name=reply]', 'hello my friend');
  await page.press('input[name=reply]', 'Enter');
  ok(await page.waitForSelector('.msg--them .now, .msg--them', { timeout: 8000 }).then(() => true).catch(() => false), 'a typed reply is answered with signs');
  await page.keyboard.press('Control+k');
  await page.fill('[role=combobox]', 'water');
  await page.keyboard.press('Enter');
  await page.waitForURL(/dictionary#water/, { timeout: 6000 }).catch(() => {});
  ok(/dictionary#water/.test(page.url()), 'the command palette jumps to /dictionary#water');
  await c.close();
}

// ---- 3. accessibility: axe in three themes ----------------------------------------------------------
section('axe');
for (const theme of ['dark', 'light', 'contrast']) {
  const c = await context(), page = await c.newPage();
  await page.addInitScript((t) => localStorage.setItem('sb.prefs', JSON.stringify({ theme: t })), theme);
  await page.goto(`${BASE}/learn`);
  await page.waitForSelector('dialog[open] input');
  await page.fill('dialog[open] input', 'Tester');
  await page.click('dialog[open] button[type=submit]');
  await page.waitForFunction(() => localStorage.getItem('sb.profile'));
  for (const path of PAGES) {
    await page.goto(BASE + path);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(700);
    await page.evaluate(AXE);
    const r = await page.evaluate(() => axe.run(document, { runOnly: ['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice'] }));
    ok(!r.violations.length, `axe ${theme} ${path}: ${r.violations.map((v) => `${v.id} (${v.nodes.length})`).join(', ')}`);
  }
  console.log(' ', theme, 'done');
  await c.close();
}

// ---- 4. phone width: no horizontal scroll ------------------------------------------------------------
section('phone');
{
  const c = await context({ viewport: { width: 390, height: 844 } }), page = await c.newPage();
  await profile(page);
  for (const path of PAGES) {
    await page.goto(BASE + path);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(400);
    ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${path} overflows at 390px`);
  }
  await c.close();
}

// ---- 5. camera speed ------------------------------------------------------------------------------------
section('speed');
{
  const c = await context(), page = await c.newPage();
  await profile(page);
  await page.goto(`${BASE}/app`);
  await page.click('button:has-text("Start camera"), button:has-text("Try again")').catch(() => {});
  await page.waitForFunction(() => window.sb?.vision?.running, null, { timeout: 60000 }).catch(() => {});
  await page.waitForTimeout(6000);
  const fps = await page.evaluate(() => window.sb?.vision?.fps ?? 0);
  console.log('  fps', fps.toFixed(1));
  ok(fps >= 15, `camera runs at ${fps.toFixed(1)} fps, wanted at least 15`);
  await c.close();
}

// ---- 6. reduced motion ------------------------------------------------------------------------------------
section('reduced motion');
{
  const c = await context({ reducedMotion: 'reduce' }), page = await c.newPage();
  await profile(page);
  await page.goto(`${BASE}/dictionary`);
  await page.click('a[href="/phrasebook"]');
  await page.waitForURL(/phrasebook/);
  await page.waitForTimeout(600);
  ok(await page.evaluate(() => document.getAnimations().filter((a) => a.playState === 'running' && a.effect?.getTiming().iterations !== Infinity).length === 0), 'no transition runs under reduced motion');
  await c.close();
}

await browser.close();
console.log(failures.length ? `\n${failures.length} failure(s)` : '\nall browser checks passed');
process.exit(failures.length ? 1 : 0);
