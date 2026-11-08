"""Render terminal transcripts (reports/evidence/<task>/*.txt) into report screenshots with headless Chrome.

    python scripts/render_transcripts.py task6 task7

Each transcript is real command output saved during the work (prompt line + command, then its output, starting with
`date`). The image gets a header that says it is a rendered transcript, so it is never mistaken for a desktop capture.
Output: reports/screenshots/<task>/<name>.png
"""

from __future__ import annotations

import html
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = """body{margin:0;background:#1e1e1e;color:#d4d4d4;font:14px/1.42 'DejaVu Sans Mono',monospace}
.bar{background:#333;color:#c8c8c8;font:12px 'DejaVu Sans',sans-serif;padding:6px 12px}
pre{margin:0;padding:10px 14px;white-space:pre-wrap;word-break:break-all}
.p{color:#4ec94e;font-weight:bold}.c{color:#e5e510;font-weight:bold}"""


def to_html(text: str, title: str) -> str:
    lines = []
    for line in text.rstrip("\n").split("\n"):
        if line.startswith("muhammad-hassan@") and "$ " in line:
            cut = line.index("$") + 1
            prompt, command = html.escape(line[:cut]), html.escape(line[cut:])
            lines.append(f'<span class="p">{prompt}</span><span class="c">{command}</span>')
        else:
            lines.append(html.escape(line))
    return (f"<!doctype html><meta charset=utf-8><style>{CSS}</style>"
            f'<div class="bar">{html.escape(title)}</div><pre>{chr(10).join(lines)}</pre>')


def render(txt: Path, out: Path, chrome: str) -> None:
    text = txt.read_text(encoding="utf-8", errors="replace")
    longest = max(len(line) for line in text.splitlines())
    rows = sum(1 + len(line) // 150 for line in text.splitlines())
    width = min(1500, max(900, int(longest * 8.5) + 40))
    height = 40 + rows * 20 + 30
    title = f"Terminal transcript (bash) · {txt.relative_to(ROOT)} · rendered from real command output"
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "t.html"
        page.write_text(to_html(text, title), encoding="utf-8")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--window-size={width},{height}", f"--screenshot={out}", page.as_uri()],
                       check=True, capture_output=True)


def main() -> None:
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        raise SystemExit("needs google-chrome or chromium")
    for task in sys.argv[1:] or ["task6"]:
        src, dst = ROOT / "reports/evidence" / task, ROOT / "reports/screenshots" / task
        dst.mkdir(parents=True, exist_ok=True)
        for txt in sorted(src.glob("*.txt")):
            render(txt, dst / f"{txt.stem}.png", chrome)
            print(f"{txt.relative_to(ROOT)} -> {(dst / (txt.stem + '.png')).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
