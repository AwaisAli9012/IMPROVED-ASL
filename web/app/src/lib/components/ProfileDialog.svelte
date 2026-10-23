<script lang="ts">
  import { getJSON, postJSON } from '../api';
  import { closeProfileDialog, profile, setProfile, type Profile } from '../stores/profile.svelte';

  let dlg: HTMLDialogElement;
  let profiles = $state<Profile[]>([]);
  let name = $state('');

  $effect(() => {
    if (profile.dialog && !dlg.open) {
      getJSON<Profile[]>('/api/profiles', { quiet: true }).then((p) => (profiles = p)).catch(() => (profiles = []));
      dlg.showModal();
    } else if (!profile.dialog && dlg.open) dlg.close();
  });

  function choose(p: Profile) {
    setProfile(p);
    closeProfileDialog();
  }

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    const n = name.trim();
    if (!n) return;
    choose(await postJSON<Profile>('/api/profiles', { name: n }));
    name = '';
  }
</script>

<dialog bind:this={dlg} aria-label="Choose a profile" oncancel={(e) => (profile.required ? e.preventDefault() : closeProfileDialog())} onclose={() => profile.dialog && closeProfileDialog()}>
  <h2>Who is learning?</h2>
  <p class="muted">Progress is saved on this computer under your name.</p>
  {#if profiles.length}
    <div class="row">
      {#each profiles as p (p.id)}
        <button class="chip chip--plain" type="button" onclick={() => choose(p)}>{p.name}</button>
      {/each}
    </div>
  {/if}
  <form class="row" onsubmit={submit}>
    <input class="field" bind:value={name} placeholder="Your name" maxlength="40" aria-label="Your name" required autocomplete="given-name" />
    <button class="btn btn--primary" type="submit">Continue</button>
  </form>
  {#if !profile.required}
    <button class="btn btn--ghost btn--sm" type="button" onclick={closeProfileDialog}>Close</button>
  {/if}
</dialog>
