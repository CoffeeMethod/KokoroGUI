# Website roadmap

Planning notes for [`docs/index.html`](index.html), the GitHub Pages landing page for the
project. This file is tracked in git, unlike the local-only `ROADMAP.md`/`PLAN_*.md` at the repo
root, because the site is a public-facing deliverable. It's fine for contributors to read where
it's headed.

## Where it stands today

Four pages, no build step: `docs/index.html` (the pitch: hero, three-step walkthrough, bento
feature grid, engine comparison table, signal chain, the `.tbaw` pitch, a changelog (4.0.0, then the README for older),
install), `docs/scripting.html` (the `[Preset:FXPreset]: Text` inline syntax, worked example
included), `docs/settings.html` (every field in every dock, including the two Audio8 fields that
are silently inert, the welcome dialog, and why JIT streaming only exists for engines that can
generate faster than real time) and `docs/format.html` (the `.tbaw` bundle: manifest keys,
`document.json` shape, segment key inputs, per-engine asset paths, the working copy, what Open
refuses). Shared tokens, nav, footer components and reference-page styles live in
`docs/assets/site.css`; `docs/assets/site.js` holds the theme toggle, the mobile menu and the
copy buttons.

The 2026-09-13 pass restyled the site along current SaaS lines: Inter and JetBrains Mono instead
of Unbounded and IBM Plex, one violet accent instead of the teal/violet/coral gradient, dark as
the default palette with light as the override, an announcement pill over a centered hero, a
framed screenshot with a glow that swaps between `shell_dark.png` and `shell_light.png` with
the theme, and a four-column footer. Page-local `<style>` blocks now hold only layouts unique
to that page; anything two pages share belongs in `site.css`.

Content was pulled from the actual code, not just `README.md`/`CLAUDE.md`. Treat "New in X.Y.Z"
entries in the README as the trigger to revisit all four pages, but check the code too; the
README's older feature-list prose has drifted before (its "adjustable interface scaling" line
still doesn't match anything in `kokoro_gui/qt/`).

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

- **A real social image.** `index.html` now carries `og:*` and `twitter:card` tags, but
  `og:image` points at `assets/shell_dark.png`, which is 1600×1000 and relative. Make a dedicated
  1200×630 image and give it an absolute URL once the Pages hostname is known.
- **Real audio samples.** The page currently only describes the difference between Kokoro and
  Audio8. A handful of short, pre-rendered `.wav`/`.mp3` clips checked into `docs/assets/audio/`,
  the same line read by a Kokoro voice, an Audio8 clone, and both generation modes, would turn the
  engine-comparison section into something a visitor can actually listen to via `<audio controls>`.
  This is the highest-impact addition on this list: a TTS project's landing page without audio
  undersells the product.
- **A second and third screenshot.** The bento cards on `index.html` fake the transcript gutter
  and timeline with CSS mockups. Real crops of the Audio FX tab and the Voice Reference dock,
  rendered by `scripts/render_screenshot.py`, would replace the two weakest ones.
- **Favicon polish.** The current favicon is a generated inline SVG. Fine for now, worth revisiting
  once there's a real logo mark.

## Phase 2: light interactivity, still no backend

Still static-hostable, no server required:

- **A tabbed signal-chain demo.** Let a visitor pick FX presets (Reverb, Compressor, Shelf
  combinations, in any mix) and hear pre-rendered before/after clips instead of just reading a
  static node chain. Keep the real order from `process_audio` as the source of truth; the demo
  should make that order audible, not reinvent it.
- **Generate the changelog.** `index.html#changelog` is hand-written from the README's "New in
  X.Y.Z" section (4.0.0 for now). A small build step that greps `README.md` into that
  `<section>` at publish time would stop it drifting, and is the point where the GitHub Actions
  workflow from option 2 above starts paying for itself.

## Phase 3: multi-page docs site

Partially done: `scripting.html`, `settings.html` and `format.html` cover the inline syntax, the
settings breakdown and the bundle format, as flat files next to `index.html` rather than a
`docs/guide/` subdirectory (not worth the extra nesting at four pages). What's left:

- Installation troubleshooting (common eSpeak NG / PyTorch setup failures and their fixes).
- A lexicon cookbook (real find-and-replace examples for acronyms, names, numbers).
- Preset-sharing conventions, if the project ever wants people to exchange `presets/*.json` files.
- Move to a static-site generator only once hand-written HTML becomes the bottleneck, not before
  (Eleventy, or plain Jekyll, which ships free on GitHub Pages with zero extra config). Four pages
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
- **An interactive timeline demo.** The app's timeline shipped in 4.0.0 and the bento card mocks
  it in CSS. A draggable in-browser version would be a second implementation of `arrangement.py`
  to keep in sync; the screenshot is enough.

## Maintenance note

Whenever the README's "New in X.Y.Z" section grows, do a pass over `docs/index.html`. The engine
comparison, bento grid, and changelog are the sections most likely to go stale first, since they
enumerate specific capabilities.

Nothing pending from the README's 4.0.0 entry: the welcome dialog and `show_welcome`
are on `settings.html`, `index.html` pitches `.tbaw` in the bento grid, the Projects section and
the changelog, and `format.html` documents the bundle. The next "New in" section reopens this
list.

Preview with `python -m http.server 8765 --directory docs` and open `http://localhost:8765/`;
opening `docs/index.html` straight from the filesystem works too, but a browser pane that
snapshots the file won't resolve `assets/`.
