#!/usr/bin/env python3
"""Validate official MoS model directories, weight shards and SHA256SUMS.

Structural checks run offline. --load additionally performs a real Gr00tPolicy
load; no fake tensor test is a substitute for that explicit acceptance step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from export_mos_release import RECIPE


COMMON = {
    "history_action_align": "shift", "history_action_offset": 1,
    "zero_init_dit": False, "confidence_tokens_enabled": False,
    "confidence_routing_enabled": False, "sub_instruction_mode": "off",
    "tune_projector": True, "tune_diffusion_model": True, "tune_vlln": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def weight_shards(model_dir: Path) -> list[Path]:
    single = model_dir / "model.safetensors"
    index = model_dir / "model.safetensors.index.json"
    if single.is_file() and index.is_file():
        raise ValueError(f"ambiguous weights in {model_dir}: both single file and index exist")
    if single.is_file():
        shards = [single]
    elif index.is_file():
        weight_map = load_json(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"missing or empty weight_map: {index}")
        root = model_dir.resolve()
        shards = []
        for name in sorted(set(weight_map.values())):
            relative = PurePosixPath(str(name))
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(name):
                raise ValueError(f"unsafe shard path in {index}: {name}")
            shard = (model_dir / Path(*relative.parts)).resolve()
            if root != shard and root not in shard.parents:
                raise ValueError(f"shard escapes model directory: {name}")
            shards.append(shard)
    else:
        raise FileNotFoundError(f"no model.safetensors or index in {model_dir}")
    for shard in shards:
        if not shard.is_file() or shard.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty weight shard: {shard}")
    return shards


def write_checksums(model_dir: Path) -> Path:
    files = sorted(p for p in model_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    if not files:
        raise ValueError(f"no files to hash under {model_dir}")
    output = model_dir / "SHA256SUMS"
    output.write_text("".join(f"{sha256(path)}  {path.relative_to(model_dir).as_posix()}\n"
                              for path in files), encoding="utf-8")
    return output


def verify_checksums(model_dir: Path) -> int:
    checksum_file = model_dir / "SHA256SUMS"
    if not checksum_file.is_file():
        raise FileNotFoundError(f"missing {checksum_file}; first run with --write-sums")
    entries = []
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or not name:
            raise ValueError(f"invalid checksum line in {checksum_file}: {line!r}")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"unsafe checksum path: {name}")
        path = (model_dir / Path(*relative.parts)).resolve()
        if model_dir.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(f"checksummed file missing or unsafe: {path}")
        if sha256(path) != digest:
            raise ValueError(f"SHA256 mismatch: {path}")
        entries.append(name)
    if not entries:
        raise ValueError(f"empty checksum manifest: {checksum_file}")
    return len(entries)


def validate_model(model_dir: Path, name: str, write_sums: bool = False) -> dict:
    if name not in RECIPE:
        raise ValueError(f"unknown official model: {name}")
    config = load_json(model_dir / "config.json")
    for key, expected in (COMMON | RECIPE[name]).items():
        if config.get(key) != expected:
            raise ValueError(f"{name}: {key}={config.get(key)!r}, expected {expected!r}")
    statistics = load_json(model_dir / "statistics.json")
    if not statistics:
        raise ValueError(f"empty statistics.json: {model_dir}")
    processor_dir = model_dir if (model_dir / "processor_config.json").is_file() else model_dir / "processor"
    processor = load_json(processor_dir / "processor_config.json")
    if not processor:
        raise ValueError(f"empty processor_config.json: {processor_dir}")
    shards = weight_shards(model_dir)
    if write_sums:
        write_checksums(model_dir)
    checksum_count = verify_checksums(model_dir)
    return {
        "name": name, "path": str(model_dir), "status": "PASS",
        "weight_shards": [str(path.relative_to(model_dir)) for path in shards],
        "weight_bytes": sum(path.stat().st_size for path in shards),
        "processor_dir": str(processor_dir.relative_to(model_dir) or Path(".")),
        "checksum_entries": checksum_count, "real_model_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=Path("release"))
    parser.add_argument("--models", nargs="+", choices=sorted(RECIPE), default=sorted(RECIPE))
    parser.add_argument("--write-sums", action="store_true",
                        help="Regenerate SHA256SUMS before verifying it")
    parser.add_argument("--load", action="store_true", help="Actually instantiate Gr00tPolicy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()
    reports = []
    for name in args.models:
        model_dir = args.release_dir / name
        report = validate_model(model_dir, name, args.write_sums)
        if args.load:
            from gr00t.policy.gr00t_policy import Gr00tPolicy
            policy = Gr00tPolicy(
                embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
                model_path=str(model_dir), device=args.device,
                denoising_steps=4, fixed_beta=0.0, fixed_noise_std=0.0,
                history_replan_stride=16,
            )
            report["real_model_loaded"] = True
            report["load_device"] = args.device
            del policy
        reports.append(report)
        print(f"PASS {model_dir}: {len(report['weight_shards'])} shard(s), "
              f"{report['checksum_entries']} hashes, real_load={report['real_model_loaded']}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({"models": reports}, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
