#!/usr/bin/env python
"""Download all pinned MoS data, verify archive hashes, and safely extract.

Every archive comes from the same immutable ModelScope dataset revision; local
staging directories are never used as an implicit substitute for public data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, spec: dict) -> None:
    actual = sha256(path)
    if actual != spec["sha256"]:
        raise ValueError(f"SHA256 mismatch for {path}: {actual}, expected {spec['sha256']}")
    if spec.get("size") is not None and path.stat().st_size != spec["size"]:
        raise ValueError(f"size mismatch for {path}: {path.stat().st_size}, expected {spec['size']}")


def safe_extract(archive: Path, output_dir: Path, dataset_dir: str) -> None:
    destination = output_dir / dataset_dir
    if (destination / "meta" / "info.json").is_file():
        print(f"Reusing extracted {destination} (archive hash already verified)")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (output_dir / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"links are not allowed in dataset archives: {member.name}")
        handle.extractall(output_dir, filter="data")
    if not (destination / "meta" / "info.json").is_file():
        raise RuntimeError(f"archive did not produce {destination}/meta/info.json")
    print(f"Extracted {archive.name} -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--cache-dir", default=".cache/mos_data")
    parser.add_argument("--revision", default=None, help="Override only for an explicit new data release")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not (root / "docs" / "DATA_MANIFEST.json").is_file():
        root = root / "MoS"  # Allow the synchronized workspace-root script copy.
    manifest = json.loads((root / "docs" / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    revision = args.revision or manifest["base_revision"]
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install ModelScope first: uv pip install modelscope") from exc
    snapshot_download(
        manifest["dataset_repo"], repo_type="dataset", revision=revision,
        allow_patterns=list(manifest["files"]), local_dir=str(cache),
    )
    for name, spec in manifest["files"].items():
        archive = cache / name
        verify_archive(archive, spec)
        safe_extract(archive, Path(args.output_dir), spec["dataset_dir"])

    from verify_mos_data import verify_dataset
    for spec in manifest["files"].values():
        report = verify_dataset(Path(args.output_dir) / spec["dataset_dir"], spec)
        print(f"PASS {report['path']}: {report['episodes']} episodes, {report['frames']} frames")


if __name__ == "__main__":
    main()
