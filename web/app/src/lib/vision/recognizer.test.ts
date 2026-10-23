import { afterEach, expect, test, vi } from 'vitest';
import { Recognizer, type Gloss } from './recognizer';
import type { VisionEvent } from './vision';

const CFG = { SEG_START_FRAMES: 3, SEG_END_FRAMES: 10, SEG_MIN_FRAMES: 12, SEG_MAX_FRAMES: 90, STATIC_STREAK: 8, SPELL_GAP_FRAMES: 25, BROW_RAISE_DELTA: 0.2, BROW_DOWN_DELTA: 0.2 };
const hand = (x: number, y: number) => Array.from({ length: 21 }, () => [x, y, 0]);
const pose = () => [[0.5, 0.2], [0.6, 0.4], [0.4, 0.4], [0.7, 0.55], [0.3, 0.55], [0.7, 0.6], [0.3, 0.6]];
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const withHand: VisionEvent = { frame: { lh: null, rh: hand(0.3, 0.6), pose: pose(), lips: null }, hand: hand(0.3, 0.6), aspect: 1.77, blend: null, pose: null };
const noHand: VisionEvent = { frame: { lh: null, rh: null, pose: pose(), lips: null }, hand: null, aspect: 1.77, blend: null, pose: null };

function fakeServer(reply: unknown) {
  const calls: { url: string; body: any }[] = [];
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => { calls.push({ url, body: JSON.parse(init.body as string) }); return { ok: true, json: async () => reply }; });
  return calls;
}
afterEach(() => vi.unstubAllGlobals());

test('a sign is sent once, after the hands rest, and accepted words become glosses', async () => {
  const calls = fakeServer({ accept: true, top: [{ label: 'hello', display: 'hello', prob: 0.9 }] });
  const glosses: Gloss[] = [];
  const rec = new Recognizer(CFG, { onGloss: (g) => glosses.push(g) });
  for (let i = 0; i < 20; i++) rec.push(withHand);
  for (let i = 0; i < 10; i++) rec.push(noHand);
  await sleep(20);
  expect(calls).toHaveLength(1);
  expect(calls[0].url).toBe('/api/predict/word');
  expect(calls[0].body.frames.length).toBeGreaterThanOrEqual(20);
  expect(glosses.map((g) => g.label)).toEqual(['hello']);
});

test('a short blip of a hand is not sent', async () => {
  const calls = fakeServer({ accept: true, top: [{ label: 'x', display: 'x', prob: 1 }] });
  const rec = new Recognizer(CFG, { onGloss: () => expect.unreachable('should not recognise a blip') });
  for (let i = 0; i < 6; i++) rec.push(withHand);
  for (let i = 0; i < 12; i++) rec.push(noHand);
  await sleep(20);
  expect(calls).toHaveLength(0);
});

test('a letter latches after a steady streak and the word is spelled once the hand leaves', async () => {
  fakeServer({ accept: true, top: [{ label: 'A', display: 'A', prob: 0.95 }] });
  const chars: string[] = [], glosses: Gloss[] = [];
  const rec = new Recognizer(CFG, { onChar: (c) => chars.push(c), onGloss: (g) => glosses.push(g) });
  rec.setMode('letters');
  for (let i = 0; i < 12; i++) { rec.push(withHand); await sleep(3); }
  expect(chars).toEqual(['A']); // held A is latched once, not once per frame
  for (let i = 0; i < 26; i++) rec.push(noHand);
  expect(glosses.map((g) => g.label)).toEqual(['fs:A']);
});
