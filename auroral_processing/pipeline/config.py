"""
YAML chain files and job files, overrides and grids.

Chain file: an ordered list of stages (or a mapping with ``label`` and ``stages``)::

    - stage: normalize
    - stage: binned_temporal_average
      bin_size: 30

Job file: which sources to process with which chains, and where to put the results::

    output: optical_flow/processed        # optional
    frames: rawimg                        # optional
    times: ut1_unix                       # optional
    preview: false                        # optional
    sources:
      - path/to/file.h5
      - dir: path/to/folder
        glob: "*.h5"
    chains:
      - chains/bin30.yaml                 # a chain file (its label is the file name)
      - label: bin10                      # or an inline chain
        stages: [...]
    grid:                                 # optional, applied to every chain that has the stage
      binned_temporal_average.bin_size: [5, 10, 30]
      wiener: [on, off]

Overrides use dotted keys: ``<stage>.<setting>`` (or ``<id>.<setting>`` for a stage with an ``id``),
and ``<stage>`` or ``<stage>.enabled`` with on/off to switch a stage on or off.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
import copy
import glob as _glob
import yaml

from auroral_processing.pipeline.processing_chain import canonical_chain, chain_json

# repo/optical_flow/processed, wherever the command is run from
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "optical_flow" / "processed"
JOB_KEYS = {"output", "frames", "times", "preview", "sources", "chains", "grid"}


def load_yaml(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def parse_value(text: str):
    """A command-line value read the way YAML would read it: "15" -> 15, "off" -> False, "null" -> None."""
    return yaml.safe_load(text) if text.strip() else text


# ---------- chains ----------
@dataclass
class ChainSpec:
    """A chain and the label to name its outputs by (None: build one from the stages)."""

    stages: list[dict]
    label: str | None = None


def load_chain(spec, base_dir=None) -> ChainSpec:
    """
    A chain from a chain file path, an inline list of stages, or a mapping
    ``{"stages": [...] or "file": path, "label": ...}``. A chain file's label defaults to its file name.
    """
    base_dir = Path(base_dir or ".")
    label = None
    if isinstance(spec, (str, Path)):
        path = _resolve(spec, base_dir)
        data = load_yaml(path)
        label = path.stem
        if isinstance(data, dict):
            label = data.get("label", label)
            data = data.get("stages")
        stages = data
    elif isinstance(spec, dict):
        if unknown := set(spec) - {"stages", "file", "label"}:
            raise ValueError(f"chain entry: unknown keys {sorted(unknown)}; use stages or file, and label")
        if "file" in spec:
            inner = load_chain(spec["file"], base_dir)
            stages, label = inner.stages, spec.get("label", inner.label)
        else:
            stages, label = spec.get("stages"), spec.get("label")
    else:
        stages = spec
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"a chain must be a non-empty list of stages, got {stages!r}")
    for i, s in enumerate(stages):
        if not isinstance(s, dict) or "stage" not in s:
            raise ValueError(f"chain entry {i} must be a mapping with a 'stage' key, got {s!r}")
    ids = [s["id"] for s in stages if "id" in s]
    if len(ids) != len(set(ids)):
        raise ValueError(f"stage ids must be unique, got {ids}")
    canonical_chain(stages)  # fail now on bad stage names or settings
    return ChainSpec(copy.deepcopy(stages), None if label is None else str(label))


def _resolve(path, base_dir) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(base_dir) / path


def _find(chain, target):
    """Indices of the stages a key's first part refers to: by id, else by stage name."""
    by_id = [i for i, s in enumerate(chain) if s.get("id") == target]
    return by_id or [i for i, s in enumerate(chain) if s["stage"] == target]


def matches(chain, key) -> bool:
    """True if override ``key`` refers to a stage in ``chain``."""
    return bool(_find(chain, key.split(".", 1)[0]))


def apply_overrides(chain, overrides: dict) -> list[dict]:
    """
    Copy of ``chain`` with dotted overrides applied, e.g. ``{"binned_temporal_average.bin_size": 15,
    "wiener": False}``. Raises if a key matches no stage, or several stages without ids.
    """
    chain = copy.deepcopy(chain)
    for key, value in overrides.items():
        target, _, setting = key.partition(".")
        idx = _find(chain, target)
        if not idx:
            raise ValueError(f"override {key!r}: no stage {target!r} in the chain "
                             f"(stages: {[s.get('id', s['stage']) for s in chain]})")
        if len(idx) > 1:
            raise ValueError(f"override {key!r}: the chain has {len(idx)} {target!r} stages; "
                             f"give them an id and use <id>.{setting or 'enabled'}")
        entry = chain[idx[0]]
        if setting in ("", "enabled"):
            entry["enabled"] = _on_off(key, value)
        elif setting in ("stage", "id"):
            raise ValueError(f"override {key!r}: can't change a stage's {setting}")
        else:
            entry[setting] = value
    canonical_chain(chain)  # fail now on unknown settings or bad values
    return chain


def _on_off(key, value) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("on", "true", "yes", "1"):
        return True
    if str(value).lower() in ("off", "false", "no", "0"):
        return False
    raise ValueError(f"override {key!r}: expected on/off, got {value!r}")


def expand_grid(chain, grid: dict) -> list[tuple[list[dict], dict]]:
    """
    Every combination of ``grid`` values applied to ``chain``, as ``(chain, overrides)`` pairs.
    Keys that refer to no stage in this chain are ignored, so one grid can serve several chains.
    Combinations that give the same chain (e.g. settings of a disabled stage) appear once.
    """
    keys = [k for k in grid if matches(chain, k)]
    out, seen = [], set()
    for values in product(*(_as_list(grid[k]) for k in keys)):
        overrides = dict(zip(keys, values))
        c = apply_overrides(chain, overrides)
        key = chain_json(canonical_chain(c))
        if key not in seen:
            seen.add(key)
            out.append((c, overrides))
    return out


def _as_list(v):
    return v if isinstance(v, list) else [v]


def override_tag(overrides: dict) -> str:
    """Short name for a set of overrides, added to explicit labels: {"wiener": False, "x.bin_size": 10} -> "nowiener_bin_size10"."""
    tags = []
    for key, value in overrides.items():
        target, _, setting = key.partition(".")
        if setting in ("", "enabled"):
            tags.append(target if _on_off(key, value) else f"no{target}")
        else:
            tags.append(f"{setting}{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool)
                        else f"{setting}{value}")
    return "_".join(tags)


def parse_assignments(items) -> dict:
    """["a.b=1", "wiener=off"] -> {"a.b": 1, "wiener": False}"""
    out = {}
    for item in items or []:
        key, eq, value = item.partition("=")
        if not eq or not key.strip():
            raise ValueError(f"bad override {item!r}; use NAME=VALUE, e.g. binned_temporal_average.bin_size=15")
        out[key.strip()] = parse_value(value)
    return out


def parse_grid(items) -> dict:
    """["a.b=1,2", "wiener=on,off"] -> {"a.b": [1, 2], "wiener": [True, False]}"""
    out = {}
    for item in items or []:
        key, eq, values = item.partition("=")
        if not eq or not key.strip() or not values:
            raise ValueError(f"bad grid {item!r}; use NAME=V1,V2, e.g. spatial_median.k=3,5")
        out[key.strip()] = [parse_value(v) for v in values.split(",")]
    return out


# ---------- sources ----------
def resolve_sources(sources, base_dir=None) -> list[Path]:
    """
    Source files from a list of entries: a file, a directory (its ``*.h5``), a glob pattern, or
    ``{"dir": folder, "glob": "*.h5"}``.
    """
    base_dir = Path(base_dir or ".")
    out = []
    for entry in sources if isinstance(sources, list) else [sources]:
        if isinstance(entry, dict):
            if unknown := set(entry) - {"dir", "glob"}:
                raise ValueError(f"source entry: unknown keys {sorted(unknown)}; use dir and glob")
            folder = _resolve(entry["dir"], base_dir)
            if not folder.is_dir():
                raise FileNotFoundError(f"source folder not found: {folder}")
            found = sorted(folder.glob(entry.get("glob", "*.h5")))
        else:
            path = _resolve(entry, base_dir)
            if path.is_dir():
                found = sorted(path.glob("*.h5"))
            elif any(c in str(entry) for c in "*?["):
                found = sorted(Path(p) for p in _glob.glob(str(path)))
            elif path.exists():
                found = [path]
            else:
                raise FileNotFoundError(f"source not found: {path}")
        if not found:
            raise FileNotFoundError(f"no source files match {entry!r}")
        out += [p for p in found if p not in out]
    return out


# ---------- jobs ----------
@dataclass
class Task:
    """One source processed by one chain."""

    source: Path
    chain: list[dict]
    label: str | None
    overrides: dict = field(default_factory=dict)


@dataclass
class Job:
    sources: list[Path]
    chains: list[ChainSpec]
    output: Path = DEFAULT_OUTPUT
    frames: str = "rawimg"
    times: str = "ut1_unix"
    grid: dict = field(default_factory=dict)
    sets: dict = field(default_factory=dict)  # overrides for every chain (--set)
    preview: bool = False

    def tasks(self) -> list[Task]:
        """sources x chains x grid combinations."""
        for what, keys in (("--set", self.sets), ("grid", self.grid)):
            if unused := [k for k in keys if not any(matches(spec.stages, k) for spec in self.chains)]:
                raise ValueError(f"{what} {unused}: no chain has that stage (check the spelling)")
        out = []
        for spec in self.chains:
            sets = {k: v for k, v in self.sets.items() if matches(spec.stages, k)}
            chain = apply_overrides(spec.stages, sets)
            for c, overrides in expand_grid(chain, self.grid):
                changed = {**sets, **overrides}
                label = spec.label
                if label and changed:  # an explicit label names the chain as written; tag what was changed
                    label = f"{label}_{override_tag(changed)}"
                out += [Task(src, c, label, changed) for src in self.sources]
        return out


def load_job(path) -> Job:
    """A job from a YAML job file. Relative paths in it are relative to the file's folder."""
    path = Path(path)
    data = load_yaml(path) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a job file must be a mapping")
    if unknown := set(data) - JOB_KEYS:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(JOB_KEYS)}")
    for key in ("sources", "chains"):
        if not data.get(key):
            raise ValueError(f"{path}: '{key}' is required")
    base = path.parent
    grid = data.get("grid") or {}
    if not isinstance(grid, dict):
        raise ValueError(f"{path}: grid must be a mapping of NAME: [values]")
    return Job(
        sources=resolve_sources(data["sources"], base),
        chains=[load_chain(c, base) for c in _as_list(data["chains"])],
        output=_resolve(data["output"], base).resolve() if data.get("output") else DEFAULT_OUTPUT,
        frames=data.get("frames", "rawimg"),
        times=data.get("times", "ut1_unix"),
        grid={k: _as_list(v) for k, v in grid.items()},
        preview=bool(data.get("preview", False)),
    )
