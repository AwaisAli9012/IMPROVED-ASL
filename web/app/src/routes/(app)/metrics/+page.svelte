<script lang="ts">
  import { onMount } from 'svelte';
  import { getJSON } from '$lib/api';
  import CountUp from '$lib/components/CountUp.svelte';
  import SortTable from '$lib/components/SortTable.svelte';

  const TASKS = [['words', 'Words'], ['static', 'Letters and numbers'], ['emotion', 'Tone from the face']] as const;
  const dec = new Intl.NumberFormat('en', { maximumFractionDigits: 3 });

  let reports = $state<Record<string, any> | null>(null);
  let tab = $state<string>('words');
  let zoom = $state<{ src: string; alt: string } | null>(null);
  let dlg: HTMLDialogElement;

  onMount(async () => { reports = await getJSON('/api/reports'); });
  $effect(() => { if (zoom && !dlg.open) dlg.showModal(); else if (!zoom && dlg.open) dlg.close(); });

  const r = $derived(reports?.[tab]);
  const title = $derived(TASKS.find(([k]) => k === tab)![1]);

  function sentence(r: any) {
    const s = r.split;
    return s.test_signers ? `${s.n_test} clips from ${s.test_signers.length} signers the model never saw, ${r.n_classes} words.` : `${s.n_test} held-out test samples, ${r.n_classes} classes.`;
  }

  /** [label, value, percent?] tiles; a null value is left out. */
  function tiles(r: any): [string, number | null, 'pct' | 'dec' | 'ms'][] {
    const t: [string, number | null, 'pct' | 'dec' | 'ms'][] = [['Accuracy', r.accuracy, 'pct'], ['Top 3', r.top3, 'pct'], ['Top 5', r.top5, 'pct'], ['Macro F1', r.f1_macro, 'pct'],
      ['Macro precision', r.precision_macro, 'pct'], ['Macro recall', r.recall_macro, 'pct'], ['Weighted F1', r.f1_weighted, 'pct'],
      ['Balanced accuracy', r.balanced_accuracy, 'pct'], ['Cohen kappa', r.cohen_kappa, 'dec'], ['MCC', r.mcc, 'dec'],
      ['Calibration error', r.ece_15bins, 'pct'], ['Speed, median', r.latency_ms_p50, 'ms'], ['Speed, 95th', r.latency_ms_p95, 'ms']];
    if (r.accept_threshold != null) t.push(['When it answers', r.precision_at_threshold, 'pct'], ['…on this share', r.coverage_at_threshold, 'pct']);
    return t.filter(([, v]) => v != null);
  }
</script>

<svelte:head><title>Metrics · SignBridge</title></svelte:head>

<section class="stack" aria-label="Model metrics">
  <div class="stack">
    <h1>How well does it work?</h1>
    <p class="muted">Every number here is measured on held-out data the models never trained on, and read from the saved evaluation reports.</p>
  </div>
  <div class="seg" role="radiogroup" aria-label="Task" style="flex-wrap: wrap">
    {#each TASKS as [k, t] (k)}<button type="button" role="radio" aria-checked={tab === k} onclick={() => (tab = k)}>{t}</button>{/each}
  </div>

  {#if !reports}
    <div class="skeleton" style="height: 18rem"></div>
  {:else if !r}
    <div class="panel"><h2>{title}</h2><p class="muted">Not trained yet. Run the training and evaluation scripts to fill this in.</p></div>
  {:else}
    <div class="panel">
      <h2>{title}</h2>
      <p class="muted">{sentence(r)}</p>
      <div class="kpis">
        {#each tiles(r) as [label, v, kind] (label)}
          <div class="kpi"><b>{#if kind === 'pct'}<CountUp value={v! * 100} digits={1} suffix="%" />{:else if kind === 'ms'}<CountUp value={v!} digits={1} suffix=" ms" />{:else}{dec.format(v!)}{/if}</b><span class="label">{label}</span></div>
        {/each}
      </div>
      {#if r.notes}<p class="muted">{r.notes}</p>{/if}
      {#if r.per_source}
        <SortTable headers={['Source', 'Accuracy', 'Macro F1', 'Samples']} rows={Object.entries<any>(r.per_source).map(([s, m]) => [s, m.accuracy, m.f1_macro, m.n])} />
      {/if}
      <div class="charts">
        {#each [['confusion', 'Confusion matrix', 'confusion matrix on the test set'], ['per_class_f1', 'F1 per class', 'F1 score for each class']] as [f, cap, alt] (f)}
          <figure style="margin: 0">
            <button class="zoom" type="button" aria-label="Open the {cap.toLowerCase()} full size" onclick={() => (zoom = { src: `/reports/${tab}_${f}.png`, alt: `${title}: ${alt}` })}>
              <img src="/reports/{tab}_{f}.png" alt="{title}: {alt}" loading="lazy" />
            </button>
            <figcaption class="label">{cap}</figcaption>
          </figure>
        {/each}
      </div>
      <SortTable filter label="Filter classes" headers={['Class', 'Precision', 'Recall', 'F1', 'Support']} rows={Object.entries<any>(r.per_class).map(([name, m]) => [name, m.precision, m.recall, m.f1, m.support])} />
      {#if r.worst_confusions?.length}
        <p class="muted">Most confused: {r.worst_confusions.slice(0, 5).map(([t, p, n]: [string, string, number]) => `${t} read as ${p} (${n})`).join(', ')}.</p>
      {/if}
    </div>
  {/if}
</section>

<dialog bind:this={dlg} class="big-chart" aria-label="Chart" onclose={() => (zoom = null)}>
  {#if zoom}<img src={zoom.src} alt={zoom.alt} />{/if}
  <button class="btn btn--sm" type="button" onclick={() => (zoom = null)}>Close</button>
</dialog>

<style>
  .zoom { display: block; width: 100%; padding: 0; border: 0; background: none; cursor: zoom-in; }
  .big-chart { max-width: 96vw; }
  .big-chart img { max-height: 80vh; width: auto; max-width: 100%; background: #fff; border-radius: var(--sc-r-md); }
</style>
