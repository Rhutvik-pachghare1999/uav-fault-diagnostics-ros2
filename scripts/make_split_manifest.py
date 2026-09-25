#!/usr/bin/env python3
"""
Freeze the run-level train/val/test split manifest (sealed protocol v2).

Protocol rules (leakage-audited):
  * Splits are made at the RUN level, stratified by fault class, BEFORE any
    augmentation. Augmented copies are later generated for TRAIN runs only.
  * Every class (16) contributes 3 train / 1 val / 2 test runs (6 runs each).
  * Train picks are forced to cover all 3 payload levels per class, so the
    training set always sees every operating point.
  * The manifest is written once and committed; every consumer (training,
    evaluation, uncertainty) must load it and never re-split on its own.

Usage:
  python3 scripts/make_split_manifest.py --out splits/split_manifest.json
"""

import argparse
import glob
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone

SEED = 42
PER_CLASS = {"train": 3, "val": 1, "test": 2}


def run_records(project_root):
    runs = sorted(glob.glob(os.path.join(project_root, "isaac_dataset", "run_*")))
    if len(runs) != 96:
        raise SystemExit(f"Expected 96 run dirs, found {len(runs)}")
    recs = {}
    for r in runs:
        name = os.path.basename(r)
        with open(os.path.join(r, "meta.json")) as f:
            m = json.load(f)
        recs[name] = {
            "class": m["fault_type"],
            "payload_kg": m.get("payload_kg"),
            "seed": m.get("seed"),
            "fault_mask": m.get("fault_mask"),
            "hover_rpm_mean": m.get("hover_rpm_mean"),
        }
    return recs


def stratified_split(recs, seed):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for name, rec in recs.items():
        by_class[rec["class"]].append(name)

    split = {"train": [], "val": [], "test": []}
    for cls in sorted(by_class):
        runs = sorted(by_class[cls])
        if len(runs) != sum(PER_CLASS.values()):
            raise SystemExit(f"class {cls}: expected 6 runs, found {len(runs)}")
        rng.shuffle(runs)
        # train: one run per payload level (covers all operating points)
        by_payload = defaultdict(list)
        for r in runs:
            by_payload[recs[r]["payload_kg"]].append(r)
        train = [rng.choice(by_payload[p]) for p in sorted(by_payload)]
        rest = [r for r in runs if r not in train]
        val = [rng.choice(rest)]
        test = [r for r in rest if r != val[0]]
        assert len(train) == 3 and len(val) == 1 and len(test) == 2
        split["train"] += train
        split["val"] += val
        split["test"] += test
    return split


def validate(split, recs):
    for part, names in split.items():
        assert len(set(names)) == len(names), f"duplicate run in {part}"
    all_runs = [n for names in split.values() for n in names]
    assert len(set(all_runs)) == 96, "splits do not cover all 96 runs exactly once"
    assert set(all_runs) == set(recs.keys())
    per_class = defaultdict(lambda: defaultdict(int))
    for part, names in split.items():
        for n in names:
            per_class[recs[n]["class"]][part] += 1
    for cls, parts in per_class.items():
        for part, want in PER_CLASS.items():
            assert parts[part] == want, f"{cls}.{part} = {parts[part]} != {want}"
    # every class present in every split
    for part in split:
        classes = {recs[n]["class"] for n in split[part]}
        assert len(classes) == 16, f"{part} missing classes: {16 - len(classes)}"
    return per_class


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out", default="splits/split_manifest.json")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    recs = run_records(args.project_root)
    split = stratified_split(recs, args.seed)
    per_class = validate(split, recs)

    def payload_of(n):
        return recs[n]["payload_kg"]

    manifest = {
        "protocol": "sealed-run-grouped-v2",
        "seed": args.seed,
        "rules": {
            "split_level": "run",
            "stratified_by": "fault class",
            "per_class": PER_CLASS,
            "augmentation": "train runs only, after split, seeded",
            "notes": "windows of a run never cross partitions; val/test are "
            "never augmented; consumers must load this file instead "
            "of re-splitting",
        },
        "runs": recs,
        "train": sorted(split["train"]),
        "val": sorted(split["val"]),
        "test": sorted(split["test"]),
        "per_class": {c: dict(p) for c, p in sorted(per_class.items())},
        "payload_coverage": {part: sorted({payload_of(n) for n in names}) for part, names in split.items()},
    }
    # sha256 pins the split CONTENT (train/val/test membership + metadata), not
    # the wall-clock "created" stamp — regenerating with the same seed must
    # reproduce the identical fingerprint, or "frozen" is a lie.
    raw = json.dumps(manifest, indent=2, sort_keys=True).encode()
    manifest["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["created"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(
        f"Wrote {args.out}: {len(split['train'])} train / {len(split['val'])} val / "
        f"{len(split['test'])} test runs (sha256 {manifest['sha256'][:12]}…)"
    )
    for cls, parts in list(per_class.items())[:3]:
        print(f"  e.g. {cls}: {dict(parts)}")


if __name__ == "__main__":
    main()
