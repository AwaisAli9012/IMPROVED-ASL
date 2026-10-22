// Settings: appearance, voice, camera, signing defaults and your data. Everything is stored with setPref().
import { download, getJSON, getPrefs, getProfile, initShell, speak, setPref, toast } from './api.js';

initShell();
const $ = (id) => document.getElementById(id);
const prefs = () => getPrefs();

// radio-style chip groups: theme, text size, default signing speed
for (const key of ['theme', 'textSize', 'avatarSpeed']) {
  const group = $(key);
  const paint = () => group.querySelectorAll('button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.value) === String(prefs()[key])));
  group.addEventListener('click', (e) => {
    if (!e.target.dataset.value) return;
    setPref(key, key === 'avatarSpeed' ? +e.target.dataset.value : e.target.dataset.value);
    paint();
  });
  paint();
}
for (const key of ['reduceMotion', 'mirror']) {
  $(key).checked = !!prefs()[key];
  $(key).addEventListener('change', (e) => setPref(key, e.target.checked));
}

// voice
const fillVoices = () => {
  const voices = speechSynthesis.getVoices().filter((v) => v.lang.startsWith('en'));
  $('voice').replaceChildren(new Option('Browser default', ''), ...voices.map((v) => new Option(`${v.name} (${v.lang})`, v.voiceURI)));
  $('voice').value = prefs().voice;
};
if ('speechSynthesis' in window) { fillVoices(); speechSynthesis.addEventListener('voiceschanged', fillVoices); } else { $('voice-note').hidden = false; $('voice').disabled = true; }
$('voice').addEventListener('change', (e) => setPref('voice', e.target.value));
$('rate').value = prefs().rate;
const showRate = () => { $('rate-out').textContent = `${(+$('rate').value).toFixed(1)}×`; };
showRate();
$('rate').addEventListener('input', () => { setPref('rate', +$('rate').value); showRate(); });
$('voice-test').addEventListener('click', () => speak('Hello, this is SignBridge.'));

// camera: names are blank until the browser has been given access once
$('camera-list').addEventListener('click', async () => {
  try {
    (await navigator.mediaDevices.getUserMedia({ video: true })).getTracks().forEach((t) => t.stop());
    const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
    $('camera').replaceChildren(new Option('Default camera', ''), ...cams.map((d, i) => new Option(d.label || `Camera ${i + 1}`, d.deviceId)));
    $('camera').value = prefs().camera;
    $('camera-note').textContent = `${cams.length} camera${cams.length === 1 ? '' : 's'} found.`;
  } catch { toast('Camera access was blocked.', 'error'); }
});
$('camera').addEventListener('change', (e) => setPref('camera', e.target.value));
if (prefs().camera) $('camera').append(new Option('Your chosen camera', prefs().camera, true, true));

// your data
$('export-data').addEventListener('click', async () => {
  const me = getProfile();
  if (!me) return toast('Open Learn and create a profile first.');
  download(`signbridge-progress-${me.name}.json`, JSON.stringify(await getJSON(`/api/progress/${me.id}`), null, 2), 'application/json');
});
$('reset-prefs').addEventListener('click', () => { try { localStorage.removeItem('sb.prefs'); } catch { /* nothing to reset */ } location.reload(); });
$('delete-profile').addEventListener('click', async () => {
  const me = getProfile();
  if (!me) return toast('There is no profile on this browser.');
  if (!confirm(`Delete "${me.name}" and all of their progress? This cannot be undone.`)) return;
  await fetch(`/api/profiles/${me.id}`, { method: 'DELETE' });
  try { localStorage.removeItem('sb.profile'); localStorage.removeItem('sb.baseline'); } catch { /* ignore */ }
  toast('Profile deleted.');
  setTimeout(() => location.assign('/'), 900);
});
