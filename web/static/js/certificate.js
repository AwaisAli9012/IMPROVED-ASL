// Certificate: printable, and only once every unit has been passed.
import { getJSON, getProfile, initShell } from './api.js';

initShell();
const profile = getProfile();
const data = profile ? await getJSON(`/api/learn/${profile.id}`).catch(() => null) : null;
if (!data?.complete) {
  document.getElementById('cert-locked').hidden = false;
} else {
  const mastered = data.units.reduce((n, u) => n + u.mastered, 0);
  document.getElementById('cert-name').textContent = data.profile.name;
  document.getElementById('cert-detail').textContent = `completed all ${data.units.length} units of the SignBridge course and mastered ${mastered} signs.`;
  document.getElementById('cert-date').textContent = new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });
  document.getElementById('cert').hidden = false;
  document.getElementById('print').addEventListener('click', () => window.print());
}
