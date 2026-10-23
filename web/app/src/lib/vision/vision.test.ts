import { expect, test } from 'vitest';
import { assignHands } from './vision';

const hand = (x: number, y: number) => Array.from({ length: 21 }, () => [x, y, 0]);
const pose = () => [[0.5, 0.2], [0.6, 0.4], [0.4, 0.4], [0.7, 0.55], [0.3, 0.55], [0.7, 0.6], [0.3, 0.6]]; // left shoulder is on the image right

test('assignHands follows the nearest wrist, drops a resting hand, and falls back without a pose', () => {
  const [l, r] = [hand(0.72, 0.6), hand(0.28, 0.6)];
  expect(assignHands([r, l], pose())).toEqual([l, r]);
  expect(assignHands([r], pose())).toEqual([null, r]);
  expect(assignHands([hand(0.3, 1.5)], pose())).toEqual([null, null]);
  const [a, b] = assignHands([hand(0.3, 0.5), hand(0.7, 0.5)], null);
  expect(a![0][0]).toBe(0.7);
  expect(b![0][0]).toBe(0.3);
});
