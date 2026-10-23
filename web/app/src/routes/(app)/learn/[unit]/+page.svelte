<script lang="ts">
  import { page } from '$app/state';
  import Lesson from '$lib/components/Lesson.svelte';
</script>

<!-- a keyed block restarts the lesson when "Next unit" changes the address -->
{#key page.params.unit}
  <Lesson unitId={page.params.unit!} />
{/key}
