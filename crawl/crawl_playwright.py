"""Bounded, source-recorded collection; robots failures are not crawl permission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from playwright.sync_api import Error, TimeoutError, sync_playwright

USER_AGENT = "PortfolioResearchCrawler/1.0"


def extract_page(page, url: str, category: str, attempts: int = 3) -> dict:
    """Retry transport/5xx failures, recording HTTP failures and extraction scope."""
    for attempt in range(1, attempts + 1):
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=20000)
            status = response.status if response else 0
            if status >= 500 or status == 429:
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 5))
                    continue
            if status >= 400 or status == 0:
                return {
                    "url": url,
                    "category": category,
                    "status": "http_error",
                    "http_status": status,
                    "attempts": attempt,
                }
            selector = "#contents"
            try:
                text = page.locator(selector).inner_text(timeout=3000)
                if len(text.strip()) < 30:
                    raise ValueError("Empty content container")
            except (TimeoutError, ValueError):
                selector = "body"
                text = page.locator(selector).inner_text(timeout=3000)
            if len(text.strip()) < 30:
                return {"url": url, "status": "empty", "attempts": attempt}
            return {
                "url": url,
                "final_url": page.url,
                "category": category,
                "title": page.title(),
                "text": text.strip(),
                "status": "ok",
                "http_status": status,
                "attempts": attempt,
                "selector": selector,
                "needs_review": selector == "body",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(text.strip().encode()).hexdigest(),
            }
        except (TimeoutError, Error) as exc:
            if attempt == attempts:
                return {
                    "url": url,
                    "status": "transport_error",
                    "attempts": attempt,
                    "error_type": type(exc).__name__,
                }
            time.sleep(min(2 ** (attempt - 1), 5))
    raise RuntimeError("Unreachable retry state")


def collect(targets: list[dict], output: Path) -> dict:
    results = []
    with sync_playwright() as runtime:
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        browser = runtime.chromium.launch(
            headless=True,
            proxy={"server": proxy, "bypass": "127.0.0.1,localhost"} if proxy else None,
        )
        context = browser.new_context(user_agent=USER_AGENT)
        policies = {}
        page = context.new_page()
        for target in targets:
            url = target["url"]
            parsed = urlsplit(url)
            if parsed.scheme not in ("https", "http") or not parsed.hostname:
                raise ValueError("Targets must be explicit HTTP(S) URLs")
            origin = f"{parsed.scheme}://{parsed.netloc}"
            if origin not in policies:
                try:
                    response = httpx.get(
                        origin + "/robots.txt",
                        timeout=15,
                        follow_redirects=True,
                        headers={"User-Agent": USER_AGENT},
                    )
                    if response.status_code != 200:
                        policies[origin] = (None, response.status_code)
                    else:
                        parser = RobotFileParser()
                        parser.parse(response.text.splitlines())
                        policies[origin] = (parser, 200)
                except httpx.HTTPError:
                    policies[origin] = (None, 0)
            policy, status = policies[origin]
            if policy is None or not policy.can_fetch(USER_AGENT, url):
                results.append(
                    {
                        "url": url,
                        "status": "robots_unverified"
                        if policy is None
                        else "robots_disallowed",
                        "robots_http_status": status,
                    }
                )
                continue
            results.append(extract_page(page, url, target.get("category", "")))
            time.sleep(max(1.5, policy.crawl_delay(USER_AGENT) or 0))
        browser.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    temporary.replace(output)
    return {
        "targets": len(targets),
        "succeeded": sum(r["status"] == "ok" for r in results),
        "statuses": {
            s: sum(r["status"] == s for r in results)
            for s in sorted({r["status"] for r in results})
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(collect(json.loads(args.targets.read_text()), args.output), indent=2)
    )


if __name__ == "__main__":
    main()
