"""Téléchargement d'images d'objets depuis Internet (Bing / DuckDuckGo via ddgs)."""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2


def object_name_to_slug(object_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", object_name.lower().strip())
    return slug.strip("_") or "objet"


def _is_valid_image_file(image_path: Path) -> bool:
    if not image_path.exists() or image_path.stat().st_size < 5000:
        return False
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    height, width = image.shape[:2]
    return height >= 64 and width >= 64


def _is_rate_limit_error(error: Exception) -> bool:
    error_name = type(error).__name__.lower()
    error_message = str(error).lower()
    return "ratelimit" in error_name or "ratelimit" in error_message or "403" in error_message


def _create_search_client():
    try:
        from ddgs import DDGS

        return DDGS(timeout=20), True
    except ImportError:
        pass

    try:
        from duckduckgo_search import DDGS

        print(
            "Note : installez le package renommé pour de meilleurs résultats : pip install ddgs"
        )
        return DDGS(), False
    except ImportError as error:
        raise ImportError(
            "Installez ddgs : pip install ddgs"
        ) from error


def _search_images(
    search_client,
    *,
    query: str,
    max_results: int,
    backend: str | None,
    supports_backend: bool,
) -> list[dict]:
    search_kwargs = {
        "query": query,
        "max_results": max_results,
        "safesearch": "off",
        "type_image": "photo",
    }
    if supports_backend and backend:
        search_kwargs["backend"] = backend

    try:
        return list(search_client.images(**search_kwargs))
    except TypeError:
        return list(search_client.images(f"{query} photo", max_results=max_results))


def _collect_search_results(
    search_client,
    *,
    query: str,
    image_count: int,
    supports_backend: bool,
) -> list[dict]:
    max_results = image_count * 4
    query_variants = [
        query,
        f"{query} photo",
        f"{query} object",
    ]
    backends = ["bing", "duckduckgo"] if supports_backend else [None]

    collected_results: list[dict] = []
    seen_urls: set[str] = set()

    for backend in backends:
        backend_label = backend or "duckduckgo"
        for query_variant in query_variants:
            for attempt in range(3):
                try:
                    batch = _search_images(
                        search_client,
                        query=query_variant,
                        max_results=max_results,
                        backend=backend,
                        supports_backend=supports_backend,
                    )
                except Exception as error:
                    if _is_rate_limit_error(error):
                        wait_seconds = 5 * (2**attempt)
                        print(
                            f"  Limite {backend_label} pour « {query_variant} », "
                            f"nouvel essai dans {wait_seconds}s..."
                        )
                        time.sleep(wait_seconds)
                        continue
                    print(f"  Recherche ignorée ({backend_label}) : {error}")
                    break

                for result in batch:
                    image_url = result.get("image") or result.get("thumbnail")
                    if image_url and image_url not in seen_urls:
                        seen_urls.add(image_url)
                        collected_results.append(result)

                if collected_results:
                    print(f"  {len(collected_results)} URL(s) via {backend_label} (« {query_variant} »)")
                    return collected_results

                time.sleep(1.5)

    return collected_results


def download_object_images(
    object_name: str,
    output_directory: Path,
    *,
    image_count: int = 10,
    search_query: str | None = None,
) -> list[Path]:
    """
    Cherche des images sur Internet et en enregistre jusqu'à image_count.
    Retourne la liste des chemins téléchargés avec succès.
    """
    import requests

    output_directory.mkdir(parents=True, exist_ok=True)
    query = search_query or object_name
    downloaded_paths: list[Path] = []

    print(f"Recherche d'images : « {query} »...")

    search_client, supports_backend = _create_search_client()
    search_results = _collect_search_results(
        search_client,
        query=query,
        image_count=image_count,
        supports_backend=supports_backend,
    )

    if not search_results:
        raise RuntimeError(
            f"Aucune image trouvée pour « {query} ».\n"
            "Essayez : --search-en, un autre nom, ou réessayez dans quelques minutes "
            "(limite DuckDuckGo/Bing)."
        )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )

    for result in search_results:
        if len(downloaded_paths) >= image_count:
            break

        image_url = result.get("image") or result.get("thumbnail")
        if not image_url:
            continue

        extension = Path(urlparse(image_url).path).suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            extension = ".jpg"

        destination_path = (
            output_directory / f"{object_name_to_slug(object_name)}_{len(downloaded_paths) + 1:02d}{extension}"
        )

        try:
            response = session.get(image_url, timeout=20)
            response.raise_for_status()
            destination_path.write_bytes(response.content)
        except requests.RequestException:
            destination_path.unlink(missing_ok=True)
            continue

        if not _is_valid_image_file(destination_path):
            destination_path.unlink(missing_ok=True)
            continue

        downloaded_paths.append(destination_path)
        print(f"  [{len(downloaded_paths)}/{image_count}] {destination_path.name}")
        time.sleep(0.3)

    if not downloaded_paths:
        raise RuntimeError(
            f"Impossible de télécharger des images valides pour « {query} ». "
            "Les URLs ont été trouvées mais les téléchargements ont échoué. Réessayez plus tard."
        )

    return downloaded_paths
