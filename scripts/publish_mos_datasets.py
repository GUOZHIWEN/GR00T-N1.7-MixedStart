#!/usr/bin/env python
"""Publish missing MoS data, pin its revision, verify downloads, then clean up.

Requires ModelScope and MODELSCOPE_API_TOKEN. Staging directories are removed
only with --remove-staged, after every public archive passes SHA256 validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from prepare_mos_data import verify_archive
from build_mos_dataset_preview import SOURCES, validate_preview


DATASET_REPO = "joyogzw/GR00T-N1.7-MixedStart-Datasets"
PREVIEW_FILE = "preview/mos_preview.parquet"
UPLOAD_FILES = ("README.md", "droid_subset_eval.tar.gz",
                "synth_mixture_eval_db7_17d.tar.gz", "droid_source_tree.json",
                PREVIEW_FILE)


def repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "docs" / "DATA_MANIFEST.json").is_file():
        root = root / "MoS"
    if not (root / "docs" / "DATA_MANIFEST.json").is_file():
        raise FileNotFoundError("cannot locate docs/DATA_MANIFEST.json")
    return root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def staged_roots(root: Path) -> list[Path]:
    roots = [root / "release" / "dataset_updates"]
    if root.name == "MoS":
        roots.append(root.parent / "release" / "dataset_updates")
    return list(dict.fromkeys(roots))


def check_staged(root: Path, manifest: dict) -> Path:
    for staging in staged_roots(root):
        if all((staging / name).is_file() for name in UPLOAD_FILES):
            for name in UPLOAD_FILES:
                if name in manifest["files"]:
                    verify_archive(staging / name, manifest["files"][name])
            validate_preview(staging / PREVIEW_FILE)
            return staging
    raise FileNotFoundError(
        f"missing complete staged upload; required: {', '.join(UPLOAD_FILES)}"
    )


def example_roots(root: Path) -> list[Path]:
    roots = [root]
    if root.name == "MoS":
        roots.append(root.parent)
    return list(dict.fromkeys(roots))


def update_example_sources(root: Path, manifest: dict) -> None:
    for candidate in example_roots(root):
        path = candidate / "examples" / "real_observation.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        archive = Path(record["source_archive"]).name
        if archive not in manifest["files"]:
            raise ValueError(f"{path}: unknown source archive {archive}")
        record.pop("local_export_source", None)
        record["source_archive"] = archive
        record["source_dataset_repo"] = manifest["dataset_repo"]
        record["source_dataset_revision"] = manifest["base_revision"]
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Updated {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=".cache/mos_data_publication")
    parser.add_argument("--remove-staged", action="store_true",
                        help="Delete both staging dirs only after all remote checks pass")
    args = parser.parse_args()
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise RuntimeError("Set MODELSCOPE_API_TOKEN; no upload was attempted")
    try:
        from modelscope.hub.api import HubApi
        from modelscope_hub.api import HubApi as UploadApi
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install ModelScope first: uv pip install modelscope") from exc

    root = repo_root()
    manifest_path = root / "docs" / "DATA_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset_repo"] != DATASET_REPO:
        raise ValueError("manifest dataset repository differs from the publication target")
    staging = check_staged(root, manifest)
    print(f"Verified staged inputs in {staging}")
    UploadApi(token=token).upload_folder(
        DATASET_REPO, repo_type="dataset", folder_path=staging,
        allow_patterns=list(UPLOAD_FILES), revision="master",
        commit_message="Add robotics classification and real tabular preview",
        commit_description="Adds explicit domain/task classification and a 40-frame "
                           "Parquet preview sampled without rescaling from held-out data.",
        tracker_path=Path(args.cache_dir) / "upload_state",
    )
    remote_files = HubApi().get_dataset_files(
        DATASET_REPO, revision="master", root_path="/", recursive=True,
    )
    records = {record["Path"]: record for record in remote_files}
    missing = sorted(set(UPLOAD_FILES) - set(records))
    if missing:
        raise RuntimeError(f"uploaded files are missing from the remote listing: {missing}")
    # Pin the commit containing the new preview, not an older archive revision.
    revision = records[PREVIEW_FILE]["Revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise RuntimeError(f"invalid remote revision: {revision!r}")
    manifest["base_revision"] = revision
    manifest["revision_note"] = (
        "All four dataset archives and the real tabular preview were downloaded from "
        "this pinned revision and verified byte-for-byte against their recorded sizes "
        "and SHA256 digests."
    )
    provenance = staging / "droid_source_tree.json"
    manifest["provenance_files"] = {
        "droid_source_tree.json": {
            "size": provenance.stat().st_size,
            "sha256": sha256(provenance),
            "source_dataset_repo": "lerobot/droid_1.0.1",
            "source_revision": "b697b419ca258bf377cbc0c3279e5348fc5c7c6f",
        }
    }
    preview = staging / PREVIEW_FILE
    manifest["preview_files"] = {
        PREVIEW_FILE: {
            "size": preview.stat().st_size,
            "sha256": sha256(preview),
            "rows": len(validate_preview(preview)),
            "purpose": "ModelScope web preview only; not a training or evaluation split",
            "sources": [name for _, name in SOURCES],
        }
    }
    cache = Path(args.cache_dir) / "download"
    snapshot_download(
        DATASET_REPO, repo_type="dataset", revision=revision,
        allow_patterns=list(manifest["files"]) + [PREVIEW_FILE],
        local_dir=str(cache), token=token,
    )
    for name, spec in manifest["files"].items():
        verify_archive(cache / name, spec)
        print(f"REMOTE PASS {name}: size and SHA256 match")
    verify_archive(cache / PREVIEW_FILE, manifest["preview_files"][PREVIEW_FILE])
    validate_preview(cache / PREVIEW_FILE, expected=preview)
    print(f"REMOTE PASS {PREVIEW_FILE}: SHA256 and all table values match")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    update_example_sources(root, manifest)
    print(f"Pinned and verified {DATASET_REPO}@{revision}")
    if args.remove_staged:
        for directory in staged_roots(root):
            if directory.is_dir():
                shutil.rmtree(directory)
                print(f"Removed verified staging directory {directory}")
    else:
        print("Staging retained. Re-run with --remove-staged to delete it after this successful verification.")


if __name__ == "__main__":
    main()
