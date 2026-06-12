#!/usr/bin/env python3
"""
Delete superseded csrc build cache entries after a new one is saved.

Called from schedule_cache_csrc_build.yaml. Environment:
    GH_TOKEN           token with actions:write on the repository
    CACHE_KEY_PREFIX   prefix shared by all generations of this cache tuple
                       (branch + os + arch + image)
    CACHE_KEY_CURRENT  the key that was just saved and must be kept
    GITHUB_REPOSITORY / GITHUB_API_URL  provided by the runner

Best-effort cleanup: every failure is reported as a warning and the script
exits 0, so pruning can never fail the build job.
"""

import json
import os
import sys
import urllib.parse
import urllib.request


def api(method: str, url: str, token: str) -> dict:
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def main() -> int:
    token = os.environ.get("GH_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    prefix = os.environ.get("CACHE_KEY_PREFIX", "")
    current = os.environ.get("CACHE_KEY_CURRENT", "")
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com")

    if not (token and repo and prefix and current):
        print("::warning::prune_csrc_caches: missing required environment; skipping")
        return 0

    query = urllib.parse.urlencode({"key": prefix, "per_page": 100})
    try:
        listing = api("GET", f"{base}/repos/{repo}/actions/caches?{query}", token)
    except Exception as e:
        print(f"::warning::prune_csrc_caches: cannot list caches: {e}")
        return 0

    deleted = 0
    for cache in listing.get("actions_caches", []):
        key = cache.get("key", "")
        cache_id = cache.get("id")
        if cache_id is None or not key.startswith(prefix) or key == current:
            continue
        try:
            api("DELETE", f"{base}/repos/{repo}/actions/caches/{cache_id}", token)
            deleted += 1
            print(f"deleted superseded cache {key} (id={cache_id}, ref={cache.get('ref')})")
        except Exception as e:
            print(f"::warning::prune_csrc_caches: cannot delete cache id={cache_id}: {e}")

    print(f"pruned {deleted} superseded cache entries for prefix {prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
