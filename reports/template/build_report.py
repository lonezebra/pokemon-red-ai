"""
Build a shareable progress-report page from the master template.

The template (progress_report_master.html) is the Game Boy-styled report
skin: DMG LCD palette, Press Start 2P display type, Gen-1 dialog boxes,
HP-bar success rates. Its content sections are ordinary HTML -- edit them
for each new report -- while every embedded asset is a __TOKEN__ this
script replaces with a base64 data URI, keeping the master editable
instead of megabytes of encoded noise.

The output is fully self-contained (the artifact CSP blocks every
external request, so fonts and images must ride along inside the file).

Usage: python3 reports/template/build_report.py
    -> writes reports/template/report.html, ready to publish.

To swap a demo GIF or add a new asset, edit ASSETS below and put a
matching __TOKEN__ in the master where the src should land. Keep an eye
on the printed final size -- the artifact limit is 16MB.
"""

import base64
import pathlib

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent.parent
SCREENSHOTS = REPO / "screenshots"

ASSETS = {
    "__FONT_PS2P__": ("font/woff2", HERE / "ps2p_latin.woff2"),
    "__GIF_ROUTE1__": ("image/gif", SCREENSHOTS / "mashups/20260727_143548/route1_mashup.gif"),
    "__GIF_ROUTE2__": ("image/gif", SCREENSHOTS / "route2_best_so_far.gif"),
    "__GIF_FOREST__": ("image/gif", SCREENSHOTS / "mashups/forest_round069/forest_mashup.gif"),
    "__GIF_ROUTE3__": ("image/gif", SCREENSHOTS / "route3_best_so_far.gif"),
    "__MAP_ROUTE3__": ("image/png", SCREENSHOTS / "route3_map.png"),
}


def main():
    html = (HERE / "progress_report_master.html").read_text()

    for token, (mime, path) in ASSETS.items():
        if token not in html:
            raise SystemExit(f"{token} not found in the master template")
        data = base64.b64encode(path.read_bytes()).decode()
        html = html.replace(token, f"data:{mime};base64,{data}")

    out = HERE / "report.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
