// The current learner. Same localStorage key as the old app.
const KEY = 'sb.profile';
export type Profile = { id: number; name: string };

function read(): Profile | null {
  try { return JSON.parse(localStorage.getItem(KEY) ?? 'null'); } catch { return null; }
}

export const profile = $state<{ current: Profile | null; dialog: boolean; required: boolean }>({ current: read(), dialog: false, required: false });

export function loadProfile() {
  profile.current = read();
}

export function setProfile(p: Profile) {
  profile.current = { id: p.id, name: p.name };
  try { localStorage.setItem(KEY, JSON.stringify(profile.current)); } catch { /* private mode: profile lasts for the page only */ }
}

let waiting: ((p: Profile | null) => void)[] = [];

/** Open the profile dialog. Resolves with the chosen profile; with `required` it cannot be dismissed. */
export function pickProfile({ required = false } = {}): Promise<Profile | null> {
  profile.required = required;
  profile.dialog = true;
  return new Promise((resolve) => waiting.push(resolve));
}

export function closeProfileDialog() {
  profile.dialog = false;
  waiting.forEach((r) => r(profile.current));
  waiting = [];
}

/** Resolve with the current profile, asking for one first when there is none. */
export async function requireProfile(): Promise<Profile> {
  return profile.current ?? (await pickProfile({ required: true }))!;
}
