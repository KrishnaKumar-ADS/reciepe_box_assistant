"""Download and safely extract the official Recipe Box archive."""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from .dataset import DATASET_FILE_GROUPS, DATASET_URL, discover_dataset_files, has_complete_dataset


def download_and_extract(data_dir: Path, progress_callback=None) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)

    # The official three JSON exports may already have been extracted by an
    # earlier downloader run. Reuse them rather than downloading 50+ MB again.
    existing = discover_dataset_files(data_dir)
    if has_complete_dataset([path.name for path in existing]):
        return [path for path in existing if path.name in {
            name for group in DATASET_FILE_GROUPS for name in group
        }]

    archive_path = data_dir / "recipes_raw.zip"
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "RecipeBoxAssistant/1.0 (dataset setup)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if progress_callback:
                progress_callback(downloaded, total)
    if not zipfile.is_zipfile(archive_path):
        archive_path.unlink(missing_ok=True)
        raise ValueError("The dataset download did not produce a valid ZIP archive.")

    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            source_name = Path(member.filename).name
            if not source_name.lower().endswith(".json"):
                continue
            # Reject path traversal and write only the basename into data/.
            if source_name in {"", ".", ".."}:
                continue
            target = data_dir / source_name
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)

    available_json = sorted(path.name for path in data_dir.glob("*.json"))
    selected: list[Path] = []
    missing_groups: list[str] = []
    for group in DATASET_FILE_GROUPS:
        match = next((data_dir / name for name in group if (data_dir / name).is_file()), None)
        if match is None:
            missing_groups.append(" or ".join(group))
        else:
            selected.append(match)
    if missing_groups:
        raise ValueError(
            "The archive was extracted, but one or more of the three Recipe Box exports "
            "could not be identified. Expected one file from each group: "
            + "; ".join(missing_groups)
            + ". JSON files found: " + (", ".join(available_json) or "none")
            + ". Check the archive contents/version."
        )
    return selected
