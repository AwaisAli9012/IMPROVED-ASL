// First-run setup shown over the live camera: confirm the camera, check framing, capture a neutral face.
import { el, refreshIcons, icon } from './api.js';
import { calibrateNeutral } from './recognizer.js';

const DONE = 'sb.onboarded';
export const needsOnboarding = () => { try { return !localStorage.getItem(DONE); } catch { return true; } };
const markDone = () => { try { localStorage.setItem(DONE, '1'); } catch { /* shown again next visit */ } };

/** Steps 2 and 3 (the camera is already on). `neutralOnly` reopens just the calibration. */
export function runOnboarding(vision, { neutralOnly = false } = {}) {
  return new Promise((resolve) => {
    const card = el('dialog', { id: 'onboard', 'aria-label': 'Set up SignBridge' });
    document.body.append(card);
    const finish = () => { markDone(); card.close(); card.remove(); resolve(); };
    const skip = el('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: () => (neutralOnly ? finish() : neutral()) }, neutralOnly ? 'Close' : 'Skip');
    const show = (...kids) => { card.replaceChildren(...kids); refreshIcons(); };
    const tick = (ok) => icon(ok ? 'circle-check' : 'circle');

    function position() {
      const seen = { shoulders: false, hand: false };
      const rows = { shoulders: el('li', { class: 'row' }), hand: el('li', { class: 'row' }) };
      const next = el('button', { class: 'btn btn--primary', type: 'button', disabled: true, onclick: () => { stop(); neutral(); } }, 'Next');
      const paint = () => {
        rows.shoulders.replaceChildren(tick(seen.shoulders), 'Both shoulders in view');
        rows.hand.replaceChildren(tick(seen.hand), 'Raise a hand');
        next.disabled = !(seen.shoulders && seen.hand);
        refreshIcons();
      };
      const stop = vision.on(({ frame }) => {
        const p = frame.pose;
        const s = !!p && [p[1], p[2]].every(([x, y]) => x > 0 && x < 1 && y > 0 && y < 1);
        const h = !!(frame.lh || frame.rh);
        if (s !== seen.shoulders || (h && !seen.hand)) { seen.shoulders = s; seen.hand ||= h; paint(); }
      });
      show(el('h2', {}, 'Check your position'), el('p', { class: 'muted' }, 'Sit so your shoulders and hands are visible, with light on your face.'),
        el('ul', { class: 'stack', style: 'list-style:none;margin:0;padding:0' }, rows.shoulders, rows.hand),
        el('div', { class: 'row' }, next, skip));
      paint();
    }

    function neutral() {
      const go = el('button', { class: 'btn btn--primary', type: 'button', onclick: async () => {
        go.disabled = true;
        go.textContent = 'Hold still…';
        const b = await calibrateNeutral(vision, 2000);
        show(el('h2', {}, b ? 'All set' : 'No face found'), el('p', { class: 'muted' }, b ? 'Your tone and question markers are now tuned to your face.' : 'Face the camera and try Recalibrate on the stage.'),
          el('button', { class: 'btn btn--primary', type: 'button', onclick: finish }, 'Start signing'));
      } }, 'Capture neutral face');
      show(el('h2', {}, 'Relax your face'), el('p', { class: 'muted' }, 'Look at the screen with a neutral expression. SignBridge compares later expressions with this one to read tone and raised eyebrows.'),
        el('div', { class: 'row' }, go, neutralOnly ? skip : el('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: finish }, 'Skip')));
    }

    card.show();
    neutralOnly ? neutral() : position();
  });
}
