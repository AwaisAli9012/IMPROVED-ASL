<script lang="ts">
  import { Tween } from 'svelte/motion';
  import { dur, ease } from '../motion';

  let { value, digits = 0, suffix = '' }: { value: number; digits?: number; suffix?: string } = $props();
  const t = new Tween(0, { duration: () => dur(400), easing: ease });
  $effect(() => { t.target = value; });
</script>

<span class="num">{t.current.toFixed(digits)}{suffix}</span>
