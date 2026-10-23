<script lang="ts">
  import '$lib/styles/tokens.css';
  import '$lib/styles/base.css';
  import '$lib/styles/shell.css';
  import { onMount } from 'svelte';
  import { applyPrefs } from '$lib/stores/prefs.svelte';
  import { loadProfile } from '$lib/stores/profile.svelte';

  let { children } = $props();
  onMount(() => {
    applyPrefs();
    loadProfile();
  });
</script>

{@render children()}
