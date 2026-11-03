"""Openverse open-licensed image retrieval (CC-BY, CC-BY-SA, CC0, Public Domain).

Openverse is an open catalogue of Creative-Commons-licensed images maintained by
the WordPress Foundation.  No API key is required for basic searches (rate-limited
to a few requests per second).

https://api.openverse.org/v1/
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_BASE = "https://api.openverse.org/v1/images/"

# Licences that are safe for an open-source educational project.
_SAFE_LICENCES = {"cc0", "pdm", "by", "by-sa"}


def search(
    prompt: str,
    *,
    license_type: str = "cc0,pdm,by,by-sa",
    per_page: int = 5,
    timeout: float = 10.0,
) -> list[dict]:
    """Return up to *per_page* open-licensed images from Openverse.

    Each result is a dict with keys ``url``, ``author``, ``license``, ``source``.
    Returns an empty list on any network or API error so callers can fall through
    to the next retrieval source.
    """
    try:
        import requests
    except ImportError:
        logger.debug("openverse: requests not available")
        return []

    try:
        resp = requests.get(
            _BASE,
            params={
                "q": prompt,
                "license_type": license_type,
                "page_size": per_page,
                "mature": "false",          # school-student safe; exclude adult content
                "extension": "jpg,png",     # avoid SVG/GIF which PIL cannot always decode
            },
            timeout=timeout,
        )
        if not resp.ok:
            logger.debug("Openverse search failed (%s) for %r", resp.status_code, prompt)
            return []
        data = resp.json()
    except Exception as exc:
        logger.debug("Openverse network error for %r: %s", prompt, exc)
        return []

    results: list[dict] = []
    for item in data.get("results", []):
        url = item.get("url", "")
        if not url or not url.startswith("http"):
            continue
        lic = (item.get("license") or "").lower().lstrip("cc-")
        if lic not in _SAFE_LICENCES:
            continue
        results.append(
            {
                "url": url,
                "author": item.get("creator") or item.get("creator_url") or "unknown",
                "license": item.get("license_version")
                    and f"CC {item['license'].upper()} {item['license_version']}"
                    or item.get("license", "").upper(),
                "source": "openverse",
                "attribution": item.get("attribution", ""),
            }
        )
    return results


def fetch_image(url: str, dest_path, timeout: float = 20.0) -> bool:
    """Download *url* to *dest_path*.  Returns True on success."""
    from pathlib import Path
    dest = Path(dest_path)
    try:
        import requests
        resp = requests.get(url, timeout=timeout, stream=True)
        if not resp.ok:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        return dest.stat().st_size > 1024      # at least 1 KB
    except Exception as exc:
        logger.debug("Openverse fetch failed for %s: %s", url, exc)
        dest.unlink(missing_ok=True)
        return False

