export type Shortcut = [key: string, what: string];

/** `page` is set by whichever page is open (Converse lists its own keys); `global` is always there. */
export const shortcuts = $state<{ open: boolean; page: Shortcut[]; global: Shortcut[] }>({
  open: false,
  page: [],
  global: [['Ctrl K', 'Search pages, signs and phrases'], ['?', 'Show this list']],
});

export const palette = $state({ open: false });
