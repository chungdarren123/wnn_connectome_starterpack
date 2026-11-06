"""Package the two train-only static attempts as submission ZIP files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/submissions/train_only_static_v1"
    output.mkdir(parents=True, exist_ok=True)
    attempts = {
        "weighted_ridge": (
            root / "solutions/train_only_static_v1/ridge/solution.py",
            root / "artifacts/train_only_static_v1/weighted_ridge.npz",
        ),
        "small_mlp": (
            root / "solutions/train_only_static_v1/mlp/solution.py",
            root / "artifacts/train_only_static_v1/small_mlp.npz",
        ),
    }
    result = {}
    for name, (solution, model) in attempts.items():
        destination = output / f"{name}.zip"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(solution, "solution.py")
            archive.write(model, "model.npz")
        with zipfile.ZipFile(destination) as archive:
            if archive.namelist() != ["solution.py", "model.npz"] or archive.testzip() is not None:
                raise ValueError(f"invalid archive {destination}")
        if destination.stat().st_size > 20 * 1024 * 1024:
            raise ValueError(f"archive exceeds 20 MiB: {destination}")
        result[name] = {
            "path": str(destination.relative_to(root)),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "contents": ["solution.py", "model.npz"],
        }
    record = root / "experiments/models/train_only_static_v1/submissions.json"
    record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
