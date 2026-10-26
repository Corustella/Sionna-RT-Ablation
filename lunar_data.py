"""Read consolidated LunarRM arrays without importing or changing its loader."""

import csv
import json
from pathlib import Path

import numpy as np


class LunarData:
    def __init__(self, root, split):
        self.root, self.split = Path(root).resolve(), split
        self.metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        index = self.root / f"{split}_index.csv"
        if not index.exists():
            index = self.root / split / "index.csv"
        with index.open(encoding="utf-8-sig", newline="") as stream:
            self.rows = list(csv.DictReader(stream))
        required = {"sample_id", "terrain_id", "hm_row", "tx_row", "tx_col"}
        if not self.rows or not required.issubset(self.rows[0]):
            raise ValueError("A consolidated LunarRM index with hm_row/TX coordinates is required")
        if len({r["sample_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Duplicate sample IDs in index")
        self.grid = tuple(self.metadata["grid"])
        self.arrays = {}

    def array_path(self, name):
        for path in (self.root / f"{self.split}_{name}.npy", self.root / self.split / f"{name}.npy"):
            if path.exists():
                return path
        raise FileNotFoundError(f"Missing consolidated {self.split}/{name}.npy")

    def array(self, name):
        if name not in self.arrays:
            self.arrays[name] = np.load(self.array_path(name), mmap_mode="r", allow_pickle=False)
        return self.arrays[name]

    def select(self, sample_ids, terrains, tx_per_terrain, seed):
        if sample_ids:
            if len(set(sample_ids)) != len(sample_ids):
                raise ValueError("Repeated --sample-ids")
            lookup = {row["sample_id"]: i for i, row in enumerate(self.rows)}
            missing = set(sample_ids) - set(lookup)
            if missing:
                raise ValueError(f"Unknown sample IDs: {sorted(missing)}")
            return [lookup[key] for key in sample_ids]
        groups = {}
        for i, row in enumerate(self.rows):
            groups.setdefault(row["terrain_id"], []).append(i)
        if not 1 <= terrains <= len(groups) or tx_per_terrain < 1:
            raise ValueError("Invalid terrain/TX selection budget")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(sorted(groups), size=terrains, replace=False)
        selected = []
        for key in sorted(chosen):
            group = groups[key]
            selected.extend(sorted(map(int, rng.choice(group, size=min(tx_per_terrain, len(group)), replace=False))))
        return selected

    def sample(self, index, bands):
        row = self.rows[index]
        height = np.array(self.array("hm")[int(row["hm_row"])], dtype=np.float64, copy=True)
        if height.shape != self.grid or not np.isfinite(height).all():
            raise ValueError(f"Invalid heightmap for {row['sample_id']}")
        tx = (int(row["tx_row"]), int(row["tx_col"]))
        if not (0 <= tx[0] < self.grid[0] and 0 <= tx[1] < self.grid[1]):
            raise ValueError("TX outside heightmap")
        result = {"height_m": height, "tx_rc": np.asarray(tx, np.int64)}
        for band in bands:
            target = np.array(self.array(f"pl{band}")[index], dtype=np.float64, copy=True)
            packed = np.asarray(self.array(f"mask{band}")[index])
            pixels = int(np.prod(self.grid))
            if target.shape != self.grid or packed.dtype != np.uint8 or packed.size != (pixels + 7) // 8:
                raise ValueError("Invalid target or bit-packed mask")
            mask = np.unpackbits(packed, count=pixels).reshape(self.grid).astype(bool)
            if not np.isfinite(target[mask]).all():
                raise ValueError("Nonfinite valid target")
            result[f"target_{band}"] = target
            result[f"mask_{band}"] = mask
        return result

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()

    def provenance(self, bands):
        paths = [self.root / "metadata.json"]
        index = self.root / f"{self.split}_index.csv"
        paths.append(index if index.exists() else self.root / self.split / "index.csv")
        paths.extend(self.array_path(field) for field in ["hm"] + [f"{p}{b}" for b in bands for p in ("pl", "mask")])
        return [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths]
