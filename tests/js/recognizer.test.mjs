// Run with: node --experimental-default-type=module --test tests/js/   (pytest does this for you)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assignHands } from '../../web/static/js/vision.js';
import { Recognizer } from '../../web/static/js/recognizer.js';

const CFG = { SEG_START_FRAMES: 3, SEG_END_FRAMES: 10, SEG_MIN_FRAMES: 12, SEG_MAX_FRAMES: 90, STATIC_STREAK: 8, SPELL_GAP_FRAMES: 25, BROW_RAISE_DELTA: 0.2, BROW_DOWN_DELTA: 0.2 };
const hand = (x, y) => Array.from({ length: 21 }, () => [x, y, 0]);
const pose = () => [[0.5, 0.2], [0.6, 0.4], [0.4, 0.4], [0.7, 0.55], [0.3, 0.55], [0.7, 0.6], [0.3, 0.6]]; // left shoulder is on the image right
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const withHand = { frame: { lh: null, rh: hand(0.3, 0.6), pose: pose(), lips: null }, hand: hand(0.3, 0.6), aspect: 1.77, blend: null };
const noHand = { frame: { lh: null, rh: null, pose: pose(), lips: null }, hand: null, aspect: 1.77, blend: null };

function fakeServer(reply) {
  const calls = [];
  globalThis.fetch = async (url, init) => { calls.push({ url, body: JSON.parse(init.body) }); return { ok: true, json: async () => reply }; };
  return calls;
}

test('assignHands follows the nearest wrist, drops a resting hand, and falls back without a pose', () => {
  const [l, r] = [hand(0.72, 0.6), hand(0.28, 0.6)];
  assert.deepEqual(assignHands([r, l], pose()), [l, r]);
  assert.deepEqual(assignHands([r], pose()), [null, r]);
  assert.deepEqual(assignHands([hand(0.3, 1.5)], pose()), [null, null]);
  const [a, b] = assignHands([hand(0.3, 0.5), hand(0.7, 0.5)], null);
  assert.equal(a[0][0], 0.7);
  assert.equal(b[0][0], 0.3);
});

test('a sign is sent once, after the hands rest, and accepted words become glosses', async () => {
  const calls = fakeServer({ accept: true, top: [{ label: 'hello', display: 'hello', prob: 0.9 }] });
  const glosses = [];
  const rec = new Recognizer(CFG, { onGloss: (g) => glosses.push(g) });
  for (let i = 0; i < 20; i++) rec.push(withHand);
  for (let i = 0; i < 10; i++) rec.push(noHand);
  await sleep(20);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/predict/word');
  assert.ok(calls[0].body.frames.length >= 20);
  assert.deepEqual(glosses.map((g) => g.label), ['hello']);
});

test('a short blip of a hand is not sent', async () => {
  const calls = fakeServer({ accept: true, top: [{ label: 'x', display: 'x', prob: 1 }] });
  const rec = new Recognizer(CFG, { onGloss: () => assert.fail('should not recognise a blip') });
  for (let i = 0; i < 6; i++) rec.push(withHand);
  for (let i = 0; i < 12; i++) rec.push(noHand);
  await sleep(20);
  assert.equal(calls.length, 0);
});

test('a letter latches after a steady streak and the word is spelled once the hand leaves', async () => {
  fakeServer({ accept: true, top: [{ label: 'A', display: 'A', prob: 0.95 }] });
  const chars = [], glosses = [];
  const rec = new Recognizer(CFG, { onChar: (c) => chars.push(c), onGloss: (g) => glosses.push(g) });
  rec.setMode('letters');
  for (let i = 0; i < 12; i++) { rec.push(withHand); await sleep(3); }
  assert.deepEqual(chars, ['A']); // held A is latched once, not once per frame
  for (let i = 0; i < 26; i++) rec.push(noHand);
  assert.deepEqual(glosses.map((g) => g.label), ['fs:A']);
});
