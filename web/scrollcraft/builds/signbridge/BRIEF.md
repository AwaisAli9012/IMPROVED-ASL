Written by the team for this project. Not based on a client interview.

# SignBridge landing page

## The eight topics (authored decisions)

1. **Vibe:** calm, warm, precise, human. References: a live-captioned theatre performance; the sign-language interpreter in the corner of a news broadcast; a field guide where every plate shows one hand shape.
2. **Journey:** you type your name and watch it signed back; you see how SignBridge reads a signer, with real measured numbers; you watch a two-way conversation happen; you see what is inside; you open it.
3. **Energy curve:** quiet and inviting at the open, rising through the numbers, loudest at the conversation, calm and direct at the close.
4. **Feeling curve** (below) and **the one moment to remember:** the conversation.
5. **One thing no site does:** the visitor's own name is fingerspelled by the signing figure, and it comes back in the conversation later.
6. **Range:** premium-minimal, product-forward. No illustration, no photography.
7. **One world or distinct scenes:** distinct scenes. Each one is the same product surface in a different state.
8. **Assets:** none generated. The page runs the real signing figure, real model reports and the real API. Reference animations are local files (see README), so on a fresh clone the figure shows its rest pose until `scripts.export_references` has run.

## Grammar: live surface

The page behaves like the product. Why the others lost: filmic one-shot needs video the product does not have; chaptered editorial and typographic poster would be a poster about a tool; gallery has nothing to browse; continuous world and split stage would invent a world the product is not in; rhythmic cutlist is for a reel. Live surface fits because the honest pitch is "watch what it does".

- App chrome replaces nav: the top strip is the real product's tab strip, linking to the real pages.
- The hero is the surface already in a state: a name already being signed.
- The close is an actual input: a name field that creates a real profile and opens the app.
- One CTA label, used once: **Open SignBridge**.

## Feeling curve

| Act | Feeling | What on screen causes it |
|---|---|---|
| Your name | Delight | A figure signs the letters of your own name, and you can step through them with the scroll wheel |
| How it reads | Trust | The method in four lines, and measured accuracy on signers the model never saw |
| The conversation (peak) | Connection | Signs become a sentence and a voice, a typed reply becomes signing, your name comes back |
| What is inside | Orientation | A plain list of the real pages with real counts |
| Open | Confidence | One field, one button |

## The peak

Sentence a visitor would tell a friend: "I watched a conversation go from hands to words and back to hands, and it signed my name."
The peak is the conversation act. It has the largest scroll span by a clear margin.

## The completed tell-someone sentence

"It's the site where your own name gets signed back to you."

## Signature move

**Your name, signed.** A name field in the hero drives the real signing figure to fingerspell it. While the hero scrolls, the wheel steps through the letters. The same name returns in the conversation.

## Authored silence

The first 18% of the conversation act is a held surface with the first message not yet shown, so the verification pass should not read it as dead scroll.

## Honesty rules in force

- No counter without a real number: every figure is read from `reports/*.json` or `/api/config` before the page mounts.
- The conversation is labelled as a scripted sample on its face. The signing in it is real animation data.
- No invented statistics, no fake dashboard, no text baked into images.
