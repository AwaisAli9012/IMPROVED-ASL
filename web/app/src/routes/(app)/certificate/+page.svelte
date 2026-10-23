<script lang="ts">
  import { onMount } from 'svelte';
  import { Printer } from '@lucide/svelte';
  import { getJSON } from '$lib/api';
  import { profile } from '$lib/stores/profile.svelte';

  let data = $state<any>(undefined); // undefined while loading, null when there is nothing to show
  onMount(async () => {
    data = profile.current ? await getJSON(`/api/learn/${profile.current.id}`).catch(() => null) : null;
  });
  const mastered = $derived(data ? data.units.reduce((n: number, u: any) => n + u.mastered, 0) : 0);
  const date = new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });
</script>

<svelte:head><title>Certificate · SignBridge</title></svelte:head>

{#if data?.complete}
  <section class="cert">
    <p class="label">SignBridge</p>
    <h1>Certificate of completion</h1>
    <p class="muted">This certifies that</p>
    <p class="big">{data.profile.name}</p>
    <p class="muted">completed all {data.units.length} units of the SignBridge course and mastered {mastered} signs.</p>
    <p class="muted num">{date}</p>
    <div class="row center no-print"><button class="btn btn--primary" type="button" onclick={() => window.print()}><Printer />Print</button></div>
  </section>
{:else if data !== undefined}
  <section class="stack">
    <h1>Not yet</h1>
    <p class="muted">Pass every unit in the course to earn your certificate.</p>
    <div class="row"><a class="btn btn--primary" href="/learn">Back to the course</a></div>
  </section>
{/if}
