# Website roadmap

Planning notes for [`docs/index.html`](index.html), the GitHub Pages landing page for the
project. This file is tracked in git, unlike the local-only `ROADMAP.md`/`PLAN_*.md` at the repo
root, because the site is a public-facing deliverable. It's fine for contributors to read where
it's headed.

## Where it stands today

Three pages, still no build step: `docs/index.html` (the pitch), `docs/scripting.html` (the
`[Preset:FXPreset]: Text` inline syntax, worked example included), and `docs/settings.html` (every
field in every dock, including the two Audio8 fields that are silently inert and why JIT streaming
only exists for engines that can generate faster than real time). Shared tokens, nav, and
components live in `docs/assets/site.css` and `docs/assets/site.js` so the three pages stay visually
consistent without copy-pasting a few hundred lines of CSS into each one.

Content was pulled from the actual code, not just `README.md`/`CLAUDE.md`, since a couple of things
those docs claim turned out to be stale (the FX chain has five groups and seventeen sliders now, not
the "Reverb/Compressor/shelf" subset CLAUDE.md still describes; the actual post-FX processing order
is Trim → Volume → Pitch → FX → Normalize, not the order this site originally shipped with; and the
README's "adjustable interface scaling and theme" line doesn't match anything in `kokoro_gui/qt/`
today, that control doesn't exist in the Qt frontend). Treat "New in X.Y.Z" entries in the README as
the trigger to revisit all three pages, the same way CLAUDE.md already asks for ROADMAP.md, but
don't assume the README's older feature-list prose is still accurate either; check the code.

## Turning the repo on for Pages

Nothing is wired up to actually serve this yet. Two options, in order of effort:

1. **Repo settings only.** Settings → Pages → Source: "Deploy from a branch" → `main` / `docs`. No
   workflow file needed, GitHub rebuilds on every push to `main` that touches `docs/`. Fastest
   option, start here.
2. **GitHub Actions workflow.** Add `.github/workflows/pages.yml` using `actions/deploy-pages` so
   the site can later include a build step (a Jekyll pass, or a bundler, if the page stops being a
   single static file). Worth doing once the site needs something option 1 can't do, not before;
   it's otherwise unnecessary CI surface.

Revisit once one of the phases below actually needs a build step.

## Phase 1: make the static page earn its keep

Low effort, no new infrastructure:

- **Open Graph and Twitter card meta tags.** `og:title`, `og:description`, `og:image`, so a link to
  the site renders a real preview card when shared instead of a bare URL. Needs a dedicated
  1200×630 social image, not a cropped screenshot.
- **Real audio samples.** The page currently only describes the difference between Kokoro and
  Audio8. A handful of short, pre-rendered `.wav`/`.mp3` clips checked into `docs/assets/audio/`,
  the same line read by a Kokoro voice, an Audio8 clone, and both generation modes, would turn the
  engine-comparison section into something a visitor can actually listen to via `<audio controls>`.
  This is the highest-impact addition on this list: a TTS project's landing page without audio
  undersells the product.
- **A second and third screenshot.** Right now there's one framed screenshot, the Generation dock.
  Add the FX dock and the Voice Reference dock so the "five docks" section has visual backing, not
  just prose. Reuse the existing `.console-frame` component.
- **Favicon polish.** The current favicon is a generated inline SVG. Fine for now, worth revisiting
  once there's a real logo mark.

## Phase 2: light interactivity, still no backend

Still static-hostable, no server required:

- **A tabbed signal-chain demo.** Let a visitor pick FX presets (Reverb, Compressor, Shelf
  combinations, in any mix) and hear pre-rendered before/after clips instead of just reading a
  static node chain. Keep the real order from `process_audio` as the source of truth; the demo
  should make that order audible, not reinvent it.
- **Copy-to-clipboard on the install code block.** Small UX win, no dependency needed since the
  Clipboard API already covers it.
- **A changelog section.** Generate it from the README's "New in X.Y.Z" headers so the site stops
  needing manual updates every release. A small build step that greps `README.md` into a
  `<section>` at publish time would do it, and this is the point where the GitHub Actions workflow
  from option 2 above starts paying for itself.

## Phase 3: multi-page docs site

Partially done: `scripting.html` and `settings.html` already cover the inline-syntax walkthrough and
the full settings breakdown, as flat files next to `index.html` rather than a `docs/guide/`
subdirectory (not worth the extra nesting at three pages). What's left:

- Installation troubleshooting (common eSpeak NG / PyTorch setup failures and their fixes).
- A lexicon cookbook (real find-and-replace examples for acronyms, names, numbers).
- Preset-sharing conventions, if the project ever wants people to exchange `presets/*.json` files.
- Move to a static-site generator only once hand-written HTML becomes the bottleneck, not before
  (Eleventy, or plain Jekyll, which ships free on GitHub Pages with zero extra config). Three pages
  sharing `assets/site.css` is still easier to keep in sync with the app than a generator would be
  at this size; revisit this once there are enough pages that the shared-CSS-file approach itself
  starts to strain.
- Versioned docs, if and when the config schema or engine abstraction changes in
  backwards-incompatible ways between releases.

## Explicitly out of scope for now

- **A live in-browser demo.** Both TTS models are multi-hundred-MB-to-multi-GB downloads that need
  PyTorch, so there's no reasonable way to run actual synthesis from a static GitHub Pages site.
  Pre-rendered audio samples (Phase 1) get most of the benefit without needing a backend, a GPU
  budget, or auth/rate-limiting. Revisit only if a hosted inference endpoint becomes something the
  project actually wants to run and pay for.
- **Analytics.** If it gets added later, keep it privacy-respecting and cookie-free (GoatCounter or
  Plausible, for example) and say so on the page. Don't add a tracker silently.
- **A waveform/timeline preview.** The app's own waveform view and multi-track timeline are still
  unbuilt (see the local planning doc for those workstreams), so the site shouldn't promise UI the
  app doesn't have yet. Add this section only after those workstreams ship.

## Maintenance note

Whenever the README's "New in X.Y.Z" section grows, do a pass over `docs/index.html`. The engine
comparison, dock grid, and feature strip are the sections most likely to go stale first, since they
enumerate specific capabilities.
