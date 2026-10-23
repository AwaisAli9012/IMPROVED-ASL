<script lang="ts">
  import { onNavigate } from '$app/navigation';
  import { page } from '$app/state';
  import { BarChart3, BookOpen, GraduationCap, MessagesSquare, Moon, Quote, Search, Settings, Sun, SunMoon, UserRound } from '@lucide/svelte';
  import CommandPalette from '$lib/components/CommandPalette.svelte';
  import ProfileDialog from '$lib/components/ProfileDialog.svelte';
  import ShortcutsSheet from '$lib/components/ShortcutsSheet.svelte';
  import Toasts from '$lib/components/Toasts.svelte';
  import { reduced } from '$lib/motion';
  import { camera } from '$lib/stores/camera.svelte';
  import { prefs } from '$lib/stores/prefs.svelte';
  import { pickProfile, profile } from '$lib/stores/profile.svelte';
  import { palette, shortcuts } from '$lib/stores/shortcuts.svelte';
  import { live } from '$lib/stores/toasts.svelte';
  import { nextTheme } from '$lib/theme';

  let { children } = $props();

  const tabs = [
    { key: '/app', label: 'Converse', icon: MessagesSquare },
    { key: '/learn', label: 'Learn', icon: GraduationCap },
    { key: '/phrasebook', label: 'Phrases', icon: Quote },
    { key: '/dictionary', label: 'Dictionary', icon: BookOpen },
    { key: '/metrics', label: 'Metrics', icon: BarChart3 },
  ];
  const active = $derived(tabs.find((t) => page.url.pathname === t.key || page.url.pathname.startsWith(t.key + '/'))?.key);
  const ThemeIcon = $derived({ dark: Moon, light: Sun, contrast: SunMoon }[prefs.theme]);

  // One underline slides between tabs. clip-path moves it, so only compositing runs.
  let nav: HTMLElement;
  $effect(() => {
    active;
    const a = nav?.querySelector<HTMLElement>('a[aria-current="page"]');
    if (!a) { nav?.style.setProperty('--ink-r', '0px'); nav?.style.setProperty('--ink-l', '0px'); return; }
    nav.style.setProperty('--ink-l', `${a.offsetLeft + 14}px`);
    nav.style.setProperty('--ink-r', `${a.offsetLeft + a.offsetWidth - 14}px`);
  });

  onNavigate((nav) => {
    if (!document.startViewTransition || reduced()) return;
    return new Promise((resolve) => {
      document.startViewTransition(async () => {
        resolve();
        await nav.complete;
      });
    });
  });

  function keydown(e: KeyboardEvent) {
    const typing = (e.target as HTMLElement).matches?.('input, textarea, select, [contenteditable]');
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); palette.open = !palette.open; }
    else if (!typing && !e.ctrlKey && !e.metaKey && !e.altKey && e.key === '?') { e.preventDefault(); shortcuts.open = true; }
    else if (!typing && !e.ctrlKey && !e.metaKey && !e.altKey && e.key === '/') { e.preventDefault(); palette.open = true; }
  }
</script>

<svelte:window onkeydown={keydown} />
<svelte:head><title>SignBridge</title></svelte:head>

<header class="topbar">
  <a class="brand" href="/" data-sveltekit-reload>SignBridge</a>
  <nav class="tabs" aria-label="Main" bind:this={nav}>
    {#each tabs as t (t.key)}
      <a href={t.key} aria-current={active === t.key ? 'page' : undefined}><t.icon />{t.label}</a>
    {/each}
    <i class="tabs__ink" aria-hidden="true"></i>
  </nav>
  <div class="topbar__right">
    <span class="dot" role="img" aria-label={{ on: 'Camera on', error: 'Camera problem', off: 'Camera off' }[camera.state]} data-state={camera.state}></span>
    <button class="btn btn--sm btn--ghost btn--icon" type="button" aria-label="Search (Ctrl K)" onclick={() => (palette.open = true)}><Search /></button>
    <button class="btn btn--sm btn--ghost btn--icon" type="button" aria-label="Switch theme, now {prefs.theme}" onclick={nextTheme}><ThemeIcon /></button>
    <a class="btn btn--sm btn--ghost btn--icon" href="/settings" aria-label="Settings"><Settings /></a>
    <button class="btn btn--sm btn--ghost" type="button" onclick={() => pickProfile()}><UserRound /><span>{profile.current?.name ?? 'Guest'}</span></button>
  </div>
</header>
<main class="page">{@render children()}</main>
<footer class="sitefoot"><a href="/about">About</a><a href="/progress">Your progress</a><a href="/captions">Live captions</a><a href="/settings">Settings</a><a href="/api/health" data-sveltekit-reload>Status</a></footer>
<div class="sr-only" aria-live="polite">{live.text}</div>
<Toasts />
<ProfileDialog />
<CommandPalette />
<ShortcutsSheet />
