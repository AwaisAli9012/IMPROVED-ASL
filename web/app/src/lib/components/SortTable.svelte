<script lang="ts">
  // A sortable table. `rows` cells are strings or numbers; numbers in every column but the last are shown as percentages.
  let { headers, rows, filter = false, label = 'Filter' }: { headers: string[]; rows: (string | number)[][]; filter?: boolean; label?: string } = $props();
  const pct = new Intl.NumberFormat('en', { style: 'percent', minimumFractionDigits: 1, maximumFractionDigits: 1 });

  let by = $state<number | null>(null);
  let desc = $state(true);
  let q = $state('');

  const view = $derived.by(() => {
    const s = q.trim().toLowerCase();
    const kept = s ? rows.filter((r) => String(r[0]).toLowerCase().includes(s)) : rows;
    if (by == null) return kept;
    const i = by;
    return [...kept].sort((a, b) => (typeof a[i] === 'string' ? (a[i] as string).localeCompare(b[i] as string) : (a[i] as number) - (b[i] as number)) * (desc ? -1 : 1));
  });

  function sort(i: number) {
    desc = by === i ? !desc : true;
    by = i;
  }
</script>

{#if filter}<input class="field" type="search" bind:value={q} placeholder={label} aria-label={label} autocomplete="off" />{/if}
<div class="table-wrap">
  <table class="table num">
    <thead><tr>
      {#each headers as h, i (h)}
        <th scope="col" tabindex="0" aria-sort={by === i ? (desc ? 'descending' : 'ascending') : 'none'} onclick={() => sort(i)} onkeydown={(e) => e.key === 'Enter' && sort(i)}>{h}</th>
      {/each}
    </tr></thead>
    <tbody>
      {#each view as row (row[0])}
        <tr>{#each row as c, i (i)}<td>{typeof c === 'number' && i < row.length - 1 ? pct.format(c) : c}</td>{/each}</tr>
      {/each}
    </tbody>
  </table>
</div>
