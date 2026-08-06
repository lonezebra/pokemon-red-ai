# Progress report template

The Game Boy-styled skin used for the project's shareable progress
reports (first published 2026-08-06). Keep the look; update the story.

## Files

- **`progress_report_master.html`** — the editable master. All the
  styling lives in its `<style>` block; the content sections below it
  are what change from report to report. Images and the pixel font
  appear as `__TOKEN__` placeholders so the master stays diffable.
- **`ps2p_latin.woff2`** — Press Start 2P (latin subset, ~5KB), the
  display face. Embedded into the page at build time because published
  artifacts cannot load external fonts.
- **`build_report.py`** — replaces every token with a base64 data URI
  and writes `report.html`, the finished self-contained page.

## Making the next report

1. Edit the content sections of `progress_report_master.html`
   (stats, checkpoints, table rows, GIF captions). The design system is
   already in the CSS: dialog boxes (`.dialog` + `bug`/`find`/`fix`),
   save-slot timeline items (`.tlitem` + `warn`/`resolved`), HP bars
   (`.hp`), spec cards, pipeline tiles.
2. Point `ASSETS` in `build_report.py` at the GIFs/screenshots this
   report should showcase (add matching tokens in the master).
3. `python3 reports/template/build_report.py`
4. Publish `report.html` (it is gitignored -- the master and script are
   the durable parts; the built page is reproducible).

## Style rules the skin commits to

- The four-shade DMG green ramp for anything that is a "screen";
  A/B-button magenta as the lone hot accent.
- Press Start 2P for display type only -- body text stays system sans.
- Square corners and hard offset shadows on game-world elements;
  both light (shell grey) and dark (backlit) themes via CSS tokens.
- Animations (blinking cursor, power LED) respect
  `prefers-reduced-motion`.
