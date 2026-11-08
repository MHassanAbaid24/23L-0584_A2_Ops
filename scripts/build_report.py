"""Build the submission PDF from report/*.md.

    python scripts/build_report.py                 # → report/23L-0584_VisionGuard.pdf
    python scripts/build_report.py --api-url https://…/api/predict --live-url https://…

Steps: render each chapter with Python-Markdown, turn image-only paragraphs into numbered <figure>s (the alt text is the
caption), build a table of contents from the chapter/section headings, fill the front-matter placeholders
({{COMMIT}}, {{BUILD_DATE}}, {{API_URL}}, {{LIVE_URL}}), then print to PDF with headless Google Chrome.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIG_RE = re.compile(r'<p>\s*<img alt="([^"]*)" src="([^"]+)"\s*/?>\s*</p>')
HEAD_RE = re.compile(r'<h([12]) id="([^"]+)">(.*?)</h\1>')


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def render(md_path: Path) -> str:
    md = markdown.Markdown(extensions=["tables", "attr_list", "fenced_code", "toc", "md_in_html"],
                           extension_configs={"toc": {"slugify": lambda v, s: md_path.stem + "-" +
                                                      re.sub(r"[^a-z0-9]+", "-", v.lower()).strip("-")}})
    body = md.convert(md_path.read_text(encoding="utf-8"))

    def figure(m: re.Match) -> str:
        alt, src = html.unescape(m.group(1)), m.group(2)
        path = (md_path.parent / src).resolve()
        if not path.exists():
            raise SystemExit(f"{md_path.name}: missing image {src}")
        label, _, rest = alt.partition(" — ")
        caption = f"<b>{html.escape(label)}.</b> {html.escape(rest)}" if rest else html.escape(alt)
        return (f'<figure><img src="{path.as_uri()}" alt="{html.escape(alt)}">'
                f"<figcaption>{caption}</figcaption></figure>")

    return FIG_RE.sub(figure, body)


def build(api_url: str, live_url: str) -> Path:
    chapters = sorted(p for p in REPORT.glob("[0-9][0-9]_*.md"))
    if not chapters:
        raise SystemExit("no report/NN_*.md chapters found")
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!report"))
    fills = {"{{COMMIT}}": commit + (" (+ uncommitted changes)" if dirty else ""),
             "{{COMMIT_SHORT}}": commit[:7], "{{BUILD_DATE}}": datetime.now().strftime("%d %B %Y"),
             "{{API_URL}}": api_url, "{{LIVE_URL}}": live_url}

    parts, toc = [], []
    for ch in chapters:
        body = render(ch)
        for key, value in fills.items():
            body = body.replace(key, html.escape(value))
        if not ch.name.startswith("00_"):
            for level, anchor, title in HEAD_RE.findall(body):
                toc.append(f'<li class="l{level}"><a href="#{anchor}">{title}</a></li>')
        parts.append(f'<section class="{"front" if ch.name.startswith("00_") else "chapter"}">{body}</section>')

    toc_html = f'<section class="toc"><h1 class="nonum">Contents</h1><ul>{"".join(toc)}</ul></section>'
    css = (REPORT / "style.css").read_text(encoding="utf-8")
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>23L-0584 VisionGuard</title>'
           f"<style>{css}</style></head><body>{parts[0]}{toc_html}{''.join(parts[1:])}</body></html>")

    out_dir = REPORT / "build"
    out_dir.mkdir(exist_ok=True)
    html_path = out_dir / "23L-0584_VisionGuard.html"
    html_path.write_text(doc, encoding="utf-8")
    pdf_path = REPORT / "23L-0584_VisionGuard.pdf"
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        raise SystemExit(f"Chrome/Chromium not found; open {html_path} in a browser and print to PDF")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   check=True, capture_output=True)
    print(f"built {pdf_path} ({pdf_path.stat().st_size / 1e6:.1f} MB) from {len(chapters)} files, commit {commit[:7]}")
    return pdf_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-url", default="https://23l-0584-a2-ops.vercel.app/api/predict (health: /api/health)")
    ap.add_argument("--live-url", default=("https://23l-0584-a2-ops.vercel.app/api/health — public model API; "
                                           "the Flask UI runs on localhost by design (Task 7) and calls it"))
    a = ap.parse_args()
    build(a.api_url, a.live_url)


if __name__ == "__main__":
    main()
