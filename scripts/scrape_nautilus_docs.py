"""
Fetch NautilusTrader documentation as raw Markdown from the GitHub source repo.

This is more reliable than scraping the rendered site (which is JS-rendered)
because the docs are maintained as .md files in the repo itself.

Source: https://github.com/nautechsystems/nautilus_trader/tree/develop/docs

Usage:
    .venv\Scripts\python.exe scripts\scrape_nautilus_docs.py [--force] [--branch develop]

Re-run any time the docs are updated to refresh the local mirror.
Output: docs/nautilustrader/<section>/<page>.md  (mirrors repo structure)
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "nautechsystems/nautilus_trader"
DEFAULT_BRANCH = "develop"
GITHUB_RAW = "https://raw.githubusercontent.com"
GITHUB_API = "https://api.github.com"
OUT_DIR = Path(__file__).parent.parent / "docs" / "nautilustrader"

# Sections to include. api_reference is large (~40 stub files); include it but
# put it last so conceptual docs are fetched first.
INCLUDE_PREFIXES = (
    "docs/getting_started",
    "docs/concepts",
    "docs/how_to",
    "docs/tutorials",
    "docs/integrations",
    "docs/developer_guide",
    "docs/api_reference",
)


def api_get(url: str) -> dict | list:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "nautilus-docs-scraper/1.0",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def raw_get(url: str) -> str | None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "nautilus-docs-scraper/1.0"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def list_doc_files(branch: str) -> list[str]:
    url = f"{GITHUB_API}/repos/{REPO}/git/trees/{branch}?recursive=1"
    data = api_get(url)
    if data.get("truncated"):
        print("[warn] GitHub tree response was truncated; some files may be missing.")
    return [
        item["path"]
        for item in data["tree"]
        if item["type"] == "blob"
        and item["path"].endswith(".md")
        and any(item["path"].startswith(p) for p in INCLUDE_PREFIXES)
    ]


def repo_path_to_local(repo_path: str) -> Path:
    # Strip leading "docs/" so local mirror is docs/nautilustrader/<rest>
    relative = repo_path.removeprefix("docs/")
    return OUT_DIR / relative


def scrape(branch: str = DEFAULT_BRANCH, force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Listing docs from {REPO}@{branch} ...")
    paths = list_doc_files(branch)
    print(f"Found {len(paths)} markdown files\n")

    saved = 0
    skipped = 0
    failed: list[str] = []

    for repo_path in paths:
        local = repo_path_to_local(repo_path)

        if local.exists() and not force:
            skipped += 1
            print(f"  [skip]  {repo_path}")
            continue

        raw_url = f"{GITHUB_RAW}/{REPO}/{branch}/{repo_path}"
        content = raw_get(raw_url)

        if content is None:
            print(f"  [FAIL]  {repo_path}")
            failed.append(repo_path)
            time.sleep(0.2)
            continue

        local.parent.mkdir(parents=True, exist_ok=True)
        # Prepend a source comment so each file is self-describing
        local.write_text(
            f"<!-- source: {raw_url} -->\n\n{content}",
            encoding="utf-8",
        )
        saved += 1
        print(f"  [ok]    {local.relative_to(OUT_DIR)}  ({len(content):,} chars)")
        time.sleep(0.05)  # polite rate; GitHub raw CDN is fast

    print(f"\nDone. {saved} saved, {skipped} skipped (already exist), {len(failed)} failed.")
    if failed:
        print("Failed:")
        for p in failed:
            print(f"  {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mirror NautilusTrader docs from GitHub to docs/nautilustrader/"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download all files even if they already exist locally.",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help=f"GitHub branch to fetch from (default: {DEFAULT_BRANCH}).",
    )
    args = parser.parse_args()
    scrape(branch=args.branch, force=args.force)
