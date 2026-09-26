# Website roadmap

Planning notes for [`docs/index.html`](index.html), the GitHub Pages landing page for the
project. This file is tracked in git, unlike the local-only `ROADMAP.md`/`PLAN_*.md` at the repo
root, because the site is a public-facing deliverable. It's fine for contributors to read where
it's headed.

## Where it stands today

Five pages, no build step. `docs/index.html` is the pitch: a hero with a transcript and its
timeline, the app screenshot, "four moves" (write, assign, generate, mix), a workflow picker
(Audiobooks, Dubbing, Podcasts & radio drama, Game & animation; tabs, the last pick remembered in
`localStorage`), the features as mixer tracks, the engine table, the signal chain as six steps,
the `.tbaw` pitch, a two-entry "What's new" and install. `docs/changes.html` is the full
changelog (the README only carries one-liners). `docs/scripting.html` is the Tags guide (the
filename stayed), `docs/settings.html` is every field in every panel, and `docs/format.html` is
the `.tbaw` spec.

The look is "script meets timeline", decided with the maintainer on 2026-09-25: the brand is
KokoroGUI with "a text-based audio workflow" as the tagline. Colors are the app's own dark and
light palettes from `kokoro_gui/qt/theme.py` and the eight character colors from
`DEFAULT_HIGHLIGHT_PALETTE` (`--c1` to `--c8`). Newsreader sets transcript text and headings,
Geist the interface, Geist Mono the gutter labels and timecode. The shared components in
`docs/assets/site.css` are `.script` (a transcript with a speaker gutter), `.run` (a
character-tinted stretch of text), `.stale` (the dashed underline), `.tl` (a timeline strip
with clips, slots and a playhead), `.tracks` (feature rows shaped like mixer tracks) and
`.eyebrow` (a gutter-style cue label). `docs/assets/site.js` adds the theme toggle, the mobile
menu, copy buttons, tab groups and the transport bar fixed to the bottom of every page: one clip
per `<main>` section, the playhead following the scroll, a clock showing the page's read-aloud
time at 150 words a minute, and Play scrolling at that pace. A section names its clip with
`data-clip` and picks a color with `data-cc`; without them the bar uses the heading and cycles
the palette. Page-local `<style>` blocks hold only layouts one page uses.

The site uses the words in `Claude/VOCABULARY.md` (Stale, Tag, FX, Mix, Reference, Export).
Where the app's own label still uses an old word ("Save FX Preset...", "Generate and render"),
the page quotes the label as it is.

`scripts/render_screenshot.py` renders `docs/assets/shell_dark.png` and `shell_light.png` with
the same Narrator / Tomas / Marta scene the hero shows.

Content was pulled from the actual code, not just `README.md`/`CLAUDE.md`. Treat a new entry on
`changes.html` as the trigger to revisit the other pages, but check the code too; feature-list
prose has drifted from the app before.

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
- **Real audio behind the hero.** The hero's transcript and timeline are HTML. Pre-rendered audio
  of that scene, played by the stage's own transport with the word highlight following it, would
  show the product doing its one trick.
- **More screenshots.** Real crops of the Audio FX tab and the Voices tab, rendered by
  `scripts/render_screenshot.py`, next to the workflow panels that talk about them.
- **Favicon polish.** The brand mark (two lines of text over a waveform) is an inline SVG. Fine
  for now, worth revisiting once there's a real logo.

## Phase 2: light interactivity, still no backend

Still static-hostable, no server required:

- **A tabbed signal-chain demo.** Let a visitor pick FX presets (Reverb, Compressor, Shelf
  combinations, in any mix) and hear pre-rendered before/after clips instead of just reading a
  static node chain. Keep the real order from `process_audio` as the source of truth; the demo
  should make that order audible, not reinvent it.
- **Check the changelog pair.** `changes.html` and the README's "What's new" are hand-written
  together. A small CI step that fails when a README bullet has no matching entry on
  `changes.html` would stop them drifting, and is the point where the GitHub Actions workflow
  from option 2 above starts paying for itself.

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
- **An interactive timeline demo.** The app's timeline shipped in 4.0.0 and the `.tl` strips mock
  it in CSS. A draggable in-browser version would be a second implementation of `arrangement.py`
  to keep in sync; the screenshot is enough.

## Maintenance note

Whenever `changes.html` grows, do a pass over `docs/index.html`. The engine comparison, the
workflow panels, the feature tracks and "What's new" are the sections most likely to go stale first, since they
enumerate specific capabilities.

The Unreleased entry on `changes.html` is covered: `settings.html` documents its settings and
menus (subprojects, subtitle import, video, convolution reverb, music beds, recordings, the
library, Fit to slot, ripple, the unified layout, the source track), `scripting.html` has the
`[pause:x]` section (07), and `format.html` has the bundle entries. Still to do: new shell
screenshots (`scripts/render_screenshot.py`), since the header column is wider and the Settings
tab grew.

Preview with `python -m http.server 8765 --directory docs` and open `http://localhost:8765/`;
opening `docs/index.html` straight from the filesystem works too, but a browser pane that
snapshots the file won't resolve `assets/`.
