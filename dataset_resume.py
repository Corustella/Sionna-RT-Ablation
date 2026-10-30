"""Explicit, narrowly scoped upgrade of the original FP32 component checkpoints."""

from datetime import datetime, timezone
import json
import sys

import numpy as np

from ablation_core import write_json

FP32_COMPONENT_COMMIT = "c0408be32d821a6ed90b99ad7d9782419e2a0fbe"


def resume_manifest(out, plan, version, runtime, upgrade_fp32=False):
    path = out / "manifest.json"
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous["experiment"] == plan and previous["code"] == version and previous["runtime"] == runtime:
        return previous
    if not upgrade_fp32:
        raise ValueError("Resume requires identical plan, data, code and RT runtime. "
                         "For the c0408be FP32 accumulation fix only, add --upgrade-fp32-accumulation")
    old_plan, new_plan = dict(previous["experiment"]), dict(plan)
    old_precision = old_plan.pop("native_gain_accumulation_dtype", "float32")
    new_precision = new_plan.pop("native_gain_accumulation_dtype", None)
    if (previous["code"] != {"commit": FP32_COMPONENT_COMMIT, "dirty": False}
            or version.get("dirty") or not version.get("commit") or previous["runtime"] != runtime
            or old_precision != "float32" or new_precision != "float64" or old_plan != new_plan
            or "accumulation_upgrade" in previous):
        raise ValueError("FP32 upgrade rejected: only clean c0408be outputs with identical "
                         "data, physical settings, seeds, batching and runtime are supported")
    # Capture precision boundaries; old arrays are never recast or recomputed.
    boundaries = {}
    for result in sorted((out / "maps").rglob("*.npz")):
        with np.load(result, allow_pickle=False) as saved:
            samples = int(saved["samples"])
            if not 0 < samples <= plan["samples_per_tx"]:
                raise ValueError(f"Invalid legacy ray count: {result}")
            if ".progress." in result.name:
                completed = int(saved["completed_batches"])
                batches = (plan["samples_per_tx"] + plan["rays_per_batch"] - 1) // plan["rays_per_batch"]
                if not 1 <= completed <= batches or samples != min(completed * plan["rays_per_batch"], plan["samples_per_tx"]):
                    raise ValueError(f"Legacy batch/ray count mismatch: {result}")
        key = result.relative_to(out).as_posix().replace(".progress.npz", ".npz")
        boundaries[key] = max(boundaries.get(key, 0), samples)
    backup = out / "manifest.before_fp64.json"
    if backup.exists():
        if json.loads(backup.read_text(encoding="utf-8")) != previous:
            raise ValueError("Existing FP32 manifest backup differs; refusing to overwrite it")
    else:
        write_json(backup, previous)
    updated = {**previous, "experiment": plan, "code": version, "runtime": runtime,
        "accumulation_upgrade": {"kind": "float32_to_float64_cell_sums_only",
            "utc": datetime.now(timezone.utc).isoformat(), "previous_code": previous["code"],
            "command": sys.argv,
            "backup": backup.name, "float32_samples_by_map": boundaries,
            "transport_weights_rng_lattice_unchanged": True}}
    write_json(path, updated)
    print(f"Upgraded manifest; preserved {len(boundaries)} FP32 results/checkpoints and {backup.name}", flush=True)
    return updated
