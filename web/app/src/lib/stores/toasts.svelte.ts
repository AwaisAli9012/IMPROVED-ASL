export type Toast = { id: number; msg: string; kind: 'info' | 'error' };

export const toasts = $state<{ list: Toast[] }>({ list: [] });
export const live = $state({ text: '' });
let next = 0;

/** Text for the polite live region that screen readers announce. */
export function announce(text: string) {
  live.text = text;
}

export function dismiss(id: number) {
  toasts.list = toasts.list.filter((t) => t.id !== id);
}

export function toast(msg: string, kind: Toast['kind'] = 'info') {
  const id = next++;
  toasts.list.push({ id, msg, kind });
  announce(msg);
  if (kind !== 'error') setTimeout(() => dismiss(id), 3600); // errors stay until dismissed
}
