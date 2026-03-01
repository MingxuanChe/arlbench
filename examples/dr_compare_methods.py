"""
Domain Randomization – BO Method Comparison Pipeline

Evaluates and compares the *HPO reliability* of different Bayesian
Optimisation algorithms (GPUCB, RAHBO, ERAHBO) by transferring their
results to a set of held-out target domains.

Evaluation protocol
-------------------
1. **Incumbents**: Each independent BO run produces one incumbent HP
   configuration.  With N runs (typically 20), we obtain N configs
   H_1, …, H_N for each method.

2. **Per-incumbent score**: For each H_i, train an RL agent *from scratch*
   directly in the target domain with K random seeds.  Average the final
   returns over seeds and target domains to obtain a scalar score S_i.

3. **HPO reliability**: Compute the mean and variance of {S_1, …, S_N}.
   - ``hpo_mean``              : E[S_i]           – expected transferred performance
   - ``hpo_std`` / ``hpo_var`` : spread of S_i    – HPO stability / reliability
   - ``hpo_mean_var_score``    : E – α·Var(S_i)   – risk-averse aggregate
   - ``hpo_worst_case_score``  : min(S_i)          – worst observed score

Selection criteria used to identify each run's incumbent:
  GPUCB  → argmax mean_performance
  RAHBO  → argmax mean_performance − alpha · Var(seed_returns)
  ERAHBO → argmax mean_performance − alpha · Var(seed_returns)

All methods share the same target domains (``--target-seed`` fixes the RNG).

Usage
-----
# Provide one run directory per method (name:path format)
python examples/dr_compare_methods.py \\
    --env-config cc_cartpole_dr --algorithm dqn \\
    --methods GPUCB:results/gpucb/run1 RAHBO:results/rahbo/run1 ERAHBO:results/erahbo/run1 \\
    --n-targets 20 --n-seeds 10

# Named shortcut flags (equivalent, for the three standard methods)
python examples/dr_compare_methods.py \\
    --env-config cc_cartpole_dr --algorithm dqn \\
    --gpucb  results/gpucb/run1 \\
    --rahbo  results/rahbo/run1 \\
    --erahbo results/erahbo/run1 \\
    --n-targets 20 --n-seeds 10

# Pass an explicit HP config YAML instead of a run directory
python examples/dr_compare_methods.py \\
    --env-config cc_cartpole_dr --algorithm dqn \\
    --methods MyMethod:hp_config.yaml OtherMethod:other.yaml \\
    --n-targets 10 --n-seeds 5

# Override variance penalty (default: alpha=1.0)
python examples/dr_compare_methods.py ... --alpha 0.5
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml

# ── allow importing from dr_target_eval.py (both live in examples/) ──────────
_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from dr_target_eval import (  # noqa: E402
    _NumpyEncoder,           # type: ignore[attr-defined]
    aggregate_results,
    build_autorl_config,
    load_env_config,
    run_target_domain,
    sample_target_domains,
)

# ─── HP-config extraction from run directories ───────────────────────────────

_CANDIDATE_FILES = [
    "merged/runhistory.csv",
    "runhistory.csv",
    "merged/incumbent.csv",
    "incumbent.csv",
]

# HP columns stored as np.log(value) in bo_trace files (natural log).
# Inverse: np.exp(stored_value) → actual HP value.
_LOG_TRANSFORMED_HP_COLS = {"learning_rate", "buffer_epsilon"}


def _parse_perf_vector(cell: str) -> np.ndarray:
    """Parse a performance vector stored as a numpy-repr string, e.g. '[1.0, 2.0]'."""
    cell = str(cell).strip()
    try:
        return np.array(ast.literal_eval(cell), dtype=float)
    except Exception:
        # Fallback: strip brackets and split
        cell = cell.lstrip("[").rstrip("]")
        parts = [p.strip() for p in cell.split(",") if p.strip()]
        return np.array([float(p) for p in parts], dtype=float)


def _resolve_run_csv(run_dir: Path) -> tuple[pd.DataFrame, bool]:
    """Find and load the runhistory/incumbent CSV from a run directory.

    Supports two layouts:
    - Standard:  ``runhistory.csv`` / ``incumbent.csv`` (or under ``merged/``)
    - BO-trace:  multiple ``bo_trace_run*.csv`` files in the directory;
                 all are concatenated so the globally best config across all
                 independent runs is available for selection.

    Returns
    -------
    (df, is_bo_trace)
        ``is_bo_trace`` is True when data came from bo_trace files, which
        means certain HP columns (``learning_rate``, ``buffer_epsilon``) are
        stored as ``np.log(value)`` and need ``np.exp()`` to recover the
        actual HP value.
    """
    # 1. Standard layout
    for rel in _CANDIDATE_FILES:
        candidate = run_dir / rel
        if candidate.exists():
            return pd.read_csv(candidate), False

    # 2. bo_trace_run*.csv layout (e.g. Efficient-Risk-Averse-BO results)
    # Search the directory itself and one level of subdirectories.
    trace_files = sorted(run_dir.glob("bo_trace_run*.csv"))
    if not trace_files:
        trace_files = sorted(run_dir.glob("*/bo_trace_run*.csv"))
    if trace_files:
        frames = [pd.read_csv(f) for f in trace_files]
        df = pd.concat(frames, ignore_index=True)
        return df, True

    raise FileNotFoundError(
        f"Could not find a runhistory/incumbent CSV or bo_trace_run*.csv files in '{run_dir}'. "
        f"Tried standard names: {_CANDIDATE_FILES}"
    )


def _strip_hp_prefix(result: tuple[pd.DataFrame, bool]) -> tuple[pd.DataFrame, bool]:
    """Rename 'hp_config.foo' columns to 'foo' for cleaner dicts."""
    df, is_bo_trace = result
    rename = {c: c.removeprefix("hp_config.") for c in df.columns if c.startswith("hp_config.")}
    return df.rename(columns=rename), is_bo_trace


def extract_best_hp_gpucb(run_dir: Path) -> dict:
    """Select the HP config that maximises mean_performance (GPUCB criterion)."""
    df, is_bo_trace = _strip_hp_prefix(_resolve_run_csv(run_dir))
    best_idx = df["mean_performance"].idxmax()
    return _row_to_hp_dict(df.iloc[best_idx], invert_log=is_bo_trace)


def extract_best_hp_rahbo(run_dir: Path, alpha: float = 1.0) -> dict:
    """Select the HP config that maximises mean - alpha * Var(seed returns).

    Requires the 'performance' column to contain the per-seed return vector.
    Falls back to mean_performance if 'performance' is missing.
    """
    df, is_bo_trace = _strip_hp_prefix(_resolve_run_csv(run_dir))
    return _best_by_mean_variance(df, alpha=alpha, invert_log=is_bo_trace)


def extract_best_hp_erahbo(run_dir: Path, alpha: float = 1.0) -> dict:
    """Select the HP config that maximises mean - alpha * Var(seed returns).

    Requires the 'performance' column to contain the per-seed return vector.
    Falls back to mean_performance if 'performance' is missing.
    """
    df, is_bo_trace = _strip_hp_prefix(_resolve_run_csv(run_dir))
    return _best_by_mean_variance(df, alpha=alpha, invert_log=is_bo_trace)


def _best_idx_by_mean_variance(
    df: pd.DataFrame,
    alpha: float,
) -> int:
    """Return the row index of the best config under mean − alpha·Var."""
    # Fast path: pre-computed columns
    if "std_performance" in df.columns and "mean_performance" in df.columns:
        means = df["mean_performance"].to_numpy(dtype=float)
        stds  = df["std_performance"].to_numpy(dtype=float)
        scores = means - alpha * stds ** 2
        return int(np.argmax(scores))

    # Slow path: parse vector string
    if "performance" in df.columns:
        scores = []
        for _, row in df.iterrows():
            perf_vec = _parse_perf_vector(row["performance"])
            scores.append(perf_vec.mean() - alpha * perf_vec.var())
        return int(np.argmax(scores))

    print(
        "  Warning: neither 'std_performance' nor 'performance' column found, "
        "falling back to mean_performance.",
        file=sys.stderr,
    )
    return int(df["mean_performance"].idxmax())


def _best_by_mean_variance(
    df: pd.DataFrame,
    alpha: float,
    invert_log: bool = False,
) -> dict:
    """Internal helper for RAHBO / ERAHBO selection (returns HP dict only)."""
    best_idx = _best_idx_by_mean_variance(df, alpha=alpha)
    return _row_to_hp_dict(df.iloc[best_idx], invert_log=invert_log)


def _cast_hp_value(v: Any) -> Any:
    """Coerce string representations of booleans / numbers to native Python types."""
    if isinstance(v, str):
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        try:
            f = float(v)
            return int(f) if f == int(f) else f
        except ValueError:
            return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _row_to_hp_dict(row: pd.Series, invert_log: bool = False) -> dict:
    """Convert a CSV row into a clean HP config dict.

    Parameters
    ----------
    invert_log:
        When True (bo_trace source), apply ``np.exp()`` to columns in
        ``_LOG_TRANSFORMED_HP_COLS`` to recover the actual HP values.
    """
    skip = {"config_id", "budget", "performance", "mean_performance",
            "std_performance", "num_samples", "mean_variance_performance"}
    result = {}
    for k, v in row.items():
        if k in skip or pd.isna(v):
            continue
        v = _cast_hp_value(v)
        if invert_log and k in _LOG_TRANSFORMED_HP_COLS and isinstance(v, (int, float)):
            v = float(np.exp(v))
        result[k] = v
    return result


def load_hp_config_from_path(path: str, method_name: str, alpha: float) -> dict:
    """Load an HP config either from a run directory or a YAML file.

    Auto-detects input type:
    - If it's a ``.yaml`` / ``.yml`` file: load directly.
    - If it's a directory: extract using the method-specific criterion.
    """
    p = Path(path)

    if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}:
        with open(p) as f:
            raw = yaml.safe_load(f)
        # Some YAMLs (e.g. Hydra algorithm configs) nest HPs under 'hp_config:'
        if isinstance(raw, dict) and "hp_config" in raw and isinstance(raw["hp_config"], dict):
            return raw["hp_config"]
        return raw

    if not p.is_dir():
        raise FileNotFoundError(f"Path '{p}' is neither a YAML file nor an existing directory.")

    name_upper = method_name.upper()
    if "GPUCB" in name_upper or "GP_UCB" in name_upper or "GP-UCB" in name_upper:
        return extract_best_hp_gpucb(p)
    elif "ERAHBO" in name_upper:
        return extract_best_hp_erahbo(p, alpha=alpha)
    elif "RAHBO" in name_upper:
        return extract_best_hp_rahbo(p, alpha=alpha)
    else:
        # Unknown name: use pure-mean criterion (most conservative choice)
        print(
            f"  Warning: unknown method name '{method_name}', selecting by max mean_performance.",
            file=sys.stderr,
        )
        return extract_best_hp_gpucb(p)


def sanitize_hp_config(hp_config: dict, algorithm: str) -> dict:
    """Clamp / round HP values to the algorithm's ConfigSpace.

    The bo_trace files may store HP values from a different version of the HP
    space (different batch-size choices, ranges, etc.).  This function maps
    each value to the nearest valid option so ARLBench's ``Configuration``
    constructor won't reject it.  Unknown keys are dropped silently.
    """
    import math
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        Constant,
        OrdinalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )
    from arlbench.core.algorithms import DQN, PPO, SAC

    alg_map = {"dqn": DQN, "ppo": PPO, "sac": SAC}
    cs = alg_map[algorithm.lower()].get_hpo_config_space()

    sanitized: dict = {}
    for hp_name, hp_obj in cs.items():
        if isinstance(hp_obj, Constant):
            sanitized[hp_name] = hp_obj.value
            continue

        raw = hp_config.get(hp_name)
        if raw is None:
            # HP not provided → use ConfigSpace default
            sanitized[hp_name] = hp_obj.default_value
            continue

        if isinstance(hp_obj, CategoricalHyperparameter):
            choices = list(hp_obj.choices)
            if raw in choices:
                # Use the canonical value from choices (not raw), to preserve the
                # correct type — e.g. int 64 rather than float 64.0.
                sanitized[hp_name] = choices[choices.index(raw)]
            else:
                # Nearest choice by numeric distance (if all choices are numeric),
                # otherwise fall back to default.
                try:
                    raw_f = float(raw)
                    nearest = min(choices, key=lambda c: abs(float(c) - raw_f))
                    sanitized[hp_name] = nearest
                except (TypeError, ValueError):
                    sanitized[hp_name] = hp_obj.default_value

        elif isinstance(hp_obj, UniformIntegerHyperparameter):
            v = int(round(float(raw)))
            sanitized[hp_name] = max(hp_obj.lower, min(hp_obj.upper, v))

        elif isinstance(hp_obj, UniformFloatHyperparameter):
            v = float(raw)
            sanitized[hp_name] = max(float(hp_obj.lower), min(float(hp_obj.upper), v))

        elif isinstance(hp_obj, OrdinalHyperparameter):
            seq = list(hp_obj.sequence)
            if raw in seq:
                sanitized[hp_name] = raw
            else:
                try:
                    raw_f = float(raw)
                    nearest = min(seq, key=lambda c: abs(float(c) - raw_f))
                    sanitized[hp_name] = nearest
                except (TypeError, ValueError):
                    sanitized[hp_name] = hp_obj.default_value
        else:
            sanitized[hp_name] = raw

    return sanitized


# ─── per-run incumbent extraction ────────────────────────────────────────────


def _extract_best_from_single_df(
    df: pd.DataFrame,
    method_name: str,
    alpha: float,
    invert_log: bool = True,
) -> tuple[dict, float]:
    """Extract the incumbent from one BO run's dataframe using the method criterion.

    Returns
    -------
    (hp_dict, source_score)
        ``source_score`` is the ``mean_performance`` of the selected incumbent
        as measured on the *source* (training) domain during BO.
    """
    rename = {c: c.removeprefix("hp_config.") for c in df.columns if c.startswith("hp_config.")}
    df = df.rename(columns=rename)
    name_upper = method_name.upper()
    if "GPUCB" in name_upper or "GP_UCB" in name_upper or "GP-UCB" in name_upper:
        best_idx = int(df["mean_performance"].idxmax())
    elif "ERAHBO" in name_upper or "RAHBO" in name_upper:
        best_idx = _best_idx_by_mean_variance(df, alpha=alpha)
    else:
        print(
            f"  Warning: unknown method '{method_name}', using max mean_performance.",
            file=sys.stderr,
        )
        best_idx = int(df["mean_performance"].idxmax())
    source_score = float(df.iloc[best_idx]["mean_performance"])
    hp = _row_to_hp_dict(df.iloc[best_idx], invert_log=invert_log)
    return hp, source_score


# ─── pkl-based incumbent extraction ──────────────────────────────────────────

def _load_pkl_data(pkl_path: Path) -> list[dict]:
    """Load ``results.pkl`` and extract the final reporting incumbent for every BO run.

    The pkl files are standard Python pickle (protocol ≥ 4) whose entries may
    contain torch Tensors; torch must therefore be importable.

    The JSON cache ``incumbent_reporting.json`` is written next to the pkl
    and invalidated whenever ``results.pkl`` is newer, so fresh BO runs
    automatically trigger re-extraction.

    Returns a list of dicts with keys:
        run_idx          : int
        incumbent_vec    : list[float]  — final reporting vector in BO space
                                         (log-scale for lr / buffer_epsilon)
        source_score     : float | None — mean score of the final incumbent
        source_score_var : float | None — return variance at the final incumbent
    """
    import pickle

    import torch  # needed to deserialise Tensor objects inside the file  # noqa: F401

    cache_path = pkl_path.parent / "incumbent_reporting.json"

    # Invalidate cache if pkl is newer
    if cache_path.exists() and cache_path.stat().st_mtime < pkl_path.stat().st_mtime:
        cache_path.unlink()

    if not cache_path.exists():
        print(f"    Loading {pkl_path} …")
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)

        data: list[dict] = []
        for i, run in enumerate(raw):
            # reporting: Tensor (n_reporting_steps, n_hp_dims) — incumbent trace in BO space
            reporting     = run["reporting"]         # final row = best HP config
            # reporting_idx: list[int] — index of each reporting point into `scores`
            reporting_idx = run["reporting_idx"]
            # scores: Tensor (n_evals, 1)  — evaluation scores at each BO query
            scores        = run["scores"]

            incumbent_vec = reporting[-1].numpy().tolist()

            # Source score: the evaluation score of the final incumbent
            best_idx = reporting_idx[-1]
            best_score_tensor = scores[best_idx]
            if torch.isfinite(best_score_tensor).any():
                source_score = float(best_score_tensor.mean())
            else:
                source_score = None

            # Source score variance: per-evaluation return variance at the same point
            scores_var_t = run.get("scores_var")
            if scores_var_t is not None:
                var_tensor = scores_var_t[best_idx]
                source_score_var = float(var_tensor.mean()) if torch.isfinite(var_tensor).any() else None
            else:
                source_score_var = None

            data.append({
                "run_idx":         i,
                "incumbent_vec":   incumbent_vec,
                "source_score":    source_score,
                "source_score_var": source_score_var,
            })

        cache_path.write_text(json.dumps(data, indent=2))
        print(f"    Cached {len(data)} run(s) → {cache_path}")

    return json.load(cache_path.open())


def _hp_names_from_trace(run_dir: Path) -> list[str]:
    """Read the HP column names (in order) from any bo_trace_run*.csv header."""
    for pat in ("bo_trace_run0.csv", "bo_trace_run*.csv", "*/bo_trace_run*.csv"):
        candidates = sorted(run_dir.glob(pat))
        if candidates:
            cols = pd.read_csv(candidates[0], nrows=0).columns.tolist()
            return [c.removeprefix("hp_config.") for c in cols if c.startswith("hp_config.")]
    raise FileNotFoundError(f"No bo_trace_run*.csv found under '{run_dir}'")


def _decode_pkl_incumbent(
    vec: list[float],
    hp_names: list[str],
    algorithm: str,
) -> dict:
    """Convert a raw reporting vector (log-space) into a sanitized HP dict.

    The reporting tensor from results.pkl is in the same transformed space
    as the bo_trace CSVs (learning_rate and buffer_epsilon are log-transformed).
    We apply np.exp() to those columns — identical to the ``invert_log=True``
    path in ``_row_to_hp_dict``.

    Transform verification
    ----------------------
    The raw ``vec`` values for ``learning_rate`` are in log-space, e.g. ≈ −8
    for a typical LR of 3e-4.  After ``np.exp()``: exp(−8) ≈ 3.4e-4 ✓.
    ``buffer_epsilon`` follows the same pattern.  All other HP dimensions are
    stored in their natural (un-transformed) units.
    """
    hp_raw: dict = {}
    for name, val in zip(hp_names, vec):
        if name in _LOG_TRANSFORMED_HP_COLS:
            hp_raw[name] = float(np.exp(val))   # invert log transform
        else:
            hp_raw[name] = _cast_hp_value(val)
    return sanitize_hp_config(hp_raw, algorithm)


def _extract_incumbents_from_pkl(
    pkl_path: Path,
    run_dir: Path,
    algorithm: str,
    output_dir: Path | None = None,
) -> tuple[list[dict], list[float]]:
    """Use results.pkl to obtain the principled BO incumbent for each run.

    The BO methods maintain a *reporting point* — the LCB/risk-averse
    maximizer — which is updated every iteration and stored in
    ``results[i]['reporting']``.  The **final row** of that tensor is
    the method's own answer to "what is the best config found in run i?".
    This is different from post-hoc argmax(mean_performance) used when
    reading bo_trace CSVs directly.

    Parameters
    ----------
    pkl_path : Path
        Path to ``results.pkl``.
    run_dir : Path
        Directory containing the bo_trace CSVs (used to read HP names and
        for the CSV-level source-score cross-check print).
    algorithm : str
        Algorithm name for ConfigSpace sanitisation.
    output_dir : Path, optional
        If given, each incumbent is saved as
        ``output_dir/incumbent_from_pkl/run_{i:02d}_incumbent.yaml``.
    """
    hp_names = _hp_names_from_trace(run_dir)
    pkl_data  = _load_pkl_data(pkl_path)

    # Also load the trace files for a cross-check of learning_rate
    trace_files = sorted(run_dir.glob("bo_trace_run*.csv"))
    if not trace_files:
        trace_files = sorted(run_dir.glob("*/bo_trace_run*.csv"))

    incumbents:        list[dict]  = []
    source_scores:     list[float] = []
    source_score_vars: list[float] = []

    n_runs = len(pkl_data)
    for entry in sorted(pkl_data, key=lambda e: e["run_idx"]):
        i       = entry["run_idx"]
        vec     = entry["incumbent_vec"]
        src     = float(entry.get("source_score") or float("nan"))
        src_var = float(entry.get("source_score_var") if entry.get("source_score_var") is not None else float("nan"))

        if vec is None:
            print(f"    run {i+1:>2}/{n_runs}: [WARN] no incumbent vector in pkl; skipping")
            continue

        hp = _decode_pkl_incumbent(vec, hp_names, algorithm)

        # ── cross-check: compare pkl LR with CSV-argmax LR ───────────────────
        lr_pkl = hp.get("learning_rate")
        if trace_files and i < len(trace_files):
            try:
                df_csv = pd.read_csv(trace_files[i])
                hp_csv, src_csv = _extract_best_from_single_df(
                    df_csv, "gpucb", alpha=1.0, invert_log=True
                )
                lr_csv = hp_csv.get("learning_rate")
                lr_note = (
                    f"  lr_pkl={lr_pkl:.2e}  lr_csv={lr_csv:.2e}"
                    if isinstance(lr_pkl, float) and isinstance(lr_csv, float)
                    else ""
                )
                if np.isnan(src):
                    src = src_csv   # fall back to CSV source score
            except Exception:
                lr_note = ""
        else:
            lr_note = f"  lr={lr_pkl:.2e}" if isinstance(lr_pkl, float) else ""

        print(f"    run {i+1:>2}/{n_runs}: pkl incumbent{lr_note}  src={src:.3f}")

        # ── save local YAML copy ──────────────────────────────────────────────
        if output_dir is not None:
            import yaml  # noqa: PLC0415
            inc_dir = output_dir / "incumbent_from_pkl"
            inc_dir.mkdir(parents=True, exist_ok=True)
            yaml_path = inc_dir / f"run_{i:02d}_incumbent.yaml"
            with open(yaml_path, "w") as fh:
                yaml.dump({"hp_config": hp, "source_score": src,
                           "source_score_var": src_var,
                           "run_idx": i, "from_pkl": str(pkl_path)}, fh)

        incumbents.append(hp)
        source_scores.append(src)
        source_score_vars.append(src_var)

    return incumbents, source_scores, source_score_vars


def extract_incumbents_per_run(
    run_dir: Path,
    algorithm: str,
    output_dir: Path | None = None,
) -> tuple[list[dict], list[float], list[float]]:
    """Extract one incumbent HP config per independent BO run from ``results.pkl``.

    Uses the BO method's own principled reporting point (LCB/risk-averse
    maximizer).  The final row of ``results[i]['reporting']`` is the method's
    recommended config after all evaluations — more correct than a post-hoc
    argmax over the bo_trace CSV.

    Transform correctness
    ---------------------
    ``results.pkl`` stores ``learning_rate`` and ``buffer_epsilon`` in
    **log-space** (natural log).  ``np.exp()`` is applied to those columns
    before passing the HP dict to any RL training call.

    Returns
    -------
    (incumbents, source_scores, source_score_vars)
        ``incumbents``         : sanitized HP config per run
        ``source_scores``      : source-domain mean return at the reporting point
        ``source_score_vars``  : source-domain return variance at the reporting point
    """
    pkl_path = run_dir / "results.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"results.pkl not found in '{run_dir}'")
    return _extract_incumbents_from_pkl(pkl_path, run_dir, algorithm, output_dir)


def load_hp_configs_for_method(
    path: str,
    method_name: str,
    alpha: float,
    algorithm: str,
) -> tuple[list[dict], list[float], list[float]]:
    """Load all HP configs for a method from ``path``.

    Returns
    -------
    (incumbents, source_scores, source_score_vars)
        - **N incumbents / N source scores / N source variances** when *path*
          is a directory with ``results.pkl`` (one entry per BO run).
        - **([single config], [nan], [nan])** when *path* is a YAML file or a
          standard run directory — source score/variance unavailable.
    """
    p = Path(path)

    # YAML file → single config, source score unknown
    if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}:
        hp = load_hp_config_from_path(path, method_name, alpha)
        hp = sanitize_hp_config(hp, algorithm)
        return [hp], [float("nan")], [float("nan")]

    if not p.is_dir():
        raise FileNotFoundError(f"Path '{p}' is neither a YAML file nor an existing directory.")

    # BO run directory with results.pkl → N incumbents (one per run)
    if (p / "results.pkl").exists():
        return extract_incumbents_per_run(p, algorithm)

    # Standard runhistory/incumbent layout → single config, source score unknown
    hp = load_hp_config_from_path(path, method_name, alpha)
    hp = sanitize_hp_config(hp, algorithm)
    return [hp], [float("nan")], [float("nan")]


def run_comparison(
    methods: list[tuple[str, str]],   # [(name, path/yaml), ...]
    env_cfg: dict,
    algorithm: str,
    target_domains: list[dict],
    seed_list: list[int],
    n_total_timesteps: float | None,
    n_eval_steps: int | None,
    n_eval_episodes: int | None,
    alpha: float,
    output_dir: Path | None,
) -> dict[str, dict]:
    """Evaluate all methods on the *same* target domains and return results.

    For each method, all N independent incumbents are evaluated on every shared
    target domain with ``len(seed_list)`` training seeds.  The resulting
    per-incumbent scalar score  S_i = mean(rewards over domains × seeds)
    is used to characterise HPO reliability:

    - ``hpo_mean``              : E[S_i]  – expected transferred performance
    - ``hpo_std`` / ``hpo_var`` : spread of S_i – HPO stability / reliability
    - ``hpo_mean_var_score``    : E[S_i] − α·Var(S_i)  (risk-averse)
    - ``hpo_worst_case_score``  : min(S_i)  (worst observed score across all incumbents)
    """
    results: dict[str, dict] = {}

    n_targets = len(target_domains)
    n_seeds = len(seed_list)

    for method_name, method_path in methods:
        print(f"\n{'─'*72}")
        print(f"  Method: {method_name}  (source: {method_path})")
        print(f"{'─'*72}")

        # ── 1. Load all incumbents ────────────────────────────────────────────
        print("  Loading incumbents ...")
        incumbents, source_scores, source_score_vars = load_hp_configs_for_method(
            method_path, method_name, alpha=alpha, algorithm=algorithm
        )
        n_inc = len(incumbents)
        print(f"  {n_inc} incumbent(s) loaded.")

        # ── 2. Evaluate every incumbent on every target domain ────────────────
        # Shape: (n_inc, n_targets, n_seeds)
        all_rewards = np.zeros((n_inc, n_targets, n_seeds), dtype=float)
        total_evals = n_inc * n_targets * n_seeds
        print(f"\n  Evaluating {n_inc} incumbents × {n_targets} domains × {n_seeds} seeds "
              f"= {total_evals} training runs ...")

        for i, hp_config in enumerate(incumbents):
            print(f"\n  Incumbent {i+1}/{n_inc}:")
            lr = hp_config.get("learning_rate")
            if isinstance(lr, float):
                print(f"    lr={lr:.2e}")

            for j, domain_params in enumerate(target_domains):
                params_str = ", ".join(f"{k}={v:.4f}" for k, v in domain_params.items())
                print(f"    domain [{j+1}/{n_targets}] {params_str}", end="", flush=True)

                inc_out_dir = (
                    output_dir / method_name / f"incumbent_{i}" if output_dir else None
                )
                rewards = run_target_domain(
                    env_cfg=env_cfg,
                    algorithm=algorithm,
                    hp_config=hp_config,
                    env_params=domain_params,
                    seed_list=seed_list,
                    n_total_timesteps=n_total_timesteps,
                    n_eval_steps=n_eval_steps,
                    n_eval_episodes=n_eval_episodes,
                    output_dir=inc_out_dir,
                )
                all_rewards[i, j, :] = rewards
                print(f"  → mean={rewards.mean():.3f}  std={rewards.std():.3f}")

        # ── 3. Compute per-incumbent scores S_i ──────────────────────────────
        # S_i = mean over all (domain, seed) pairs for incumbent i
        incumbent_scores = all_rewards.mean(axis=(1, 2))  # (n_inc,)

        source_arr      = np.array(source_scores,      dtype=float)
        source_vars_arr = np.array(source_score_vars, dtype=float)

        results[method_name] = {
            "incumbents":           incumbents,
            "n_incumbents":         n_inc,
            "all_rewards":          all_rewards,           # (n_inc, n_targets, n_seeds)
            "incumbent_scores":     incumbent_scores,      # (n_inc,)
            "source_scores":        source_arr,            # (n_inc,) – source-domain mean perf
            "source_score_vars":    source_vars_arr,       # (n_inc,) – source-domain Var
            "hpo_mean":             float(incumbent_scores.mean()),
            "hpo_std":              float(incumbent_scores.std()),
            "hpo_var":              float(incumbent_scores.var()),
            "hpo_min":              float(incumbent_scores.min()),
            "hpo_max":              float(incumbent_scores.max()),
            "hpo_mean_var_score":   float(
                incumbent_scores.mean() - alpha * incumbent_scores.var()
            ),
            "hpo_worst_case_score": float(incumbent_scores.min()),
        }

    return results


# ─── reporting ────────────────────────────────────────────────────────────────


def print_comparison(
    results: dict[str, dict],
    target_domains: list[dict],
    alpha: float,
) -> None:
    """Print side-by-side HPO reliability comparison table.

    For each method the table shows statistics of the per-incumbent scores
    S_i (mean over all target domains and seeds for incumbent i):

      hpo_mean              E[S_i]          – expected transferred performance
      hpo_std / hpo_var     spread of S_i   – HPO stability
      hpo_mean_var_score    E[S_i] − α·Var  – risk-averse aggregate
      hpo_worst_case_score  min(S_i)        – worst observed score
    """
    sep = "=" * 80
    print(f"\n{sep}")
    print("  DR Evaluation – HPO Reliability Comparison")
    print(sep)

    methods = list(results.keys())
    col_w = 18

    n_inc_per_method = {m: results[m]["n_incumbents"] for m in methods}
    multi_inc = any(n > 1 for n in n_inc_per_method.values())
    if multi_inc:
        inc_str = "  ".join(f"{m}={n}" for m, n in n_inc_per_method.items())
        print(f"\n  Incumbents per method:  {inc_str}")

    # ── HPO reliability table ──────────────────────────────────────────────────
    col_w = 18
    header = f"  {'Metric':<35}" + "".join(f"{m:>{col_w}}" for m in methods)
    if multi_inc:
        print("\n[HPO Reliability]  (statistics over N independent BO incumbents)")
    else:
        print("\n[Performance Metrics]")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows_def = [
        ("E[S_i]  mean perf.",             "hpo_mean",             ".4f"),
        ("Std[S_i]  HPO instability",       "hpo_std",              ".4f"),
        ("Var[S_i]  HPO variance",          "hpo_var",              ".4f"),
        (f"E[S_i] − α·Var  (α={alpha})",   "hpo_mean_var_score",   ".4f"),
        ("Worst-case  min(S_i)",             "hpo_worst_case_score", ".4f"),
        ("Min S_i",                         "hpo_min",              ".4f"),
        ("Max S_i",                         "hpo_max",              ".4f"),
    ]

    for label, key, fmt in rows_def:
        row_str = f"  {label:<35}"
        for m in methods:
            val = results[m].get(key, float("nan"))
            row_str += f"{val:>{col_w}{fmt}}"
        print(row_str)

    # ── Per-domain mean (averaged over all incumbents) ─────────────────────────
    n_targets = len(target_domains)
    print("\n[Per-Domain Mean Performance  (averaged over all incumbents)]")
    dom_header = f"  {'Domain':>6}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(dom_header)
    print("  " + "-" * (len(dom_header) - 2))

    for j in range(n_targets):
        row_str = f"  {j:>6d}"
        for m in methods:
            # all_rewards: (n_inc, n_targets, n_seeds)
            val = float(results[m]["all_rewards"][:, j, :].mean())
            row_str += f"{val:>{col_w}.4f}"
        print(row_str)

    # ── Per-incumbent scores S_i (only when multi-incumbent) ──────────────────
    if multi_inc:
        max_n = max(n_inc_per_method.values())
        print("\n[Per-Incumbent Scores S_i  (mean over all domains × seeds)]")
        inc_header = f"  {'Incumbent':>10}" + "".join(f"{m:>{col_w}}" for m in methods)
        print(inc_header)
        print("  " + "-" * (len(inc_header) - 2))
        for i in range(max_n):
            row_str = f"  {i+1:>10d}"
            for m in methods:
                scores = results[m]["incumbent_scores"]
                val = float(scores[i]) if i < len(scores) else float("nan")
                row_str += f"{val:>{col_w}.4f}"
            print(row_str)

    print(sep)


def save_comparison(
    output_dir: Path,
    results: dict[str, dict],
    target_domains: list[dict],
    seed_list: list[int],
    env_cfg: dict,
    algorithm: str,
    alpha: float,
) -> None:
    """Save all comparison outputs to ``output_dir``.

    File layout
    -----------
    <output_dir>/
        comparison_summary.csv      – headline HPO reliability stats per method
        incumbent_scores.csv        – S_i for every method × incumbent
        per_domain_comparison.csv   – per-domain means averaged over incumbents
        target_domains.json         – shared target domain parameters
        <method>/
            incumbent_scores.csv    – S_i per incumbent for this method
            incumbent_<i>/
                hp_config.yaml
                results.csv         – raw (domain, seed, reward) triples
                per_domain_summary.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(results.keys())

    # ── 1. Comparison summary CSV (one row per method) ─────────────────────────
    rows = []
    for m in methods:
        r = results[m]
        rows.append({
            "method":               m,
            "n_incumbents":         r["n_incumbents"],
            "hpo_mean":             r["hpo_mean"],
            "hpo_std":              r["hpo_std"],
            "hpo_var":              r["hpo_var"],
            "hpo_min":              r["hpo_min"],
            "hpo_max":              r["hpo_max"],
            "hpo_mean_var_score":   r["hpo_mean_var_score"],
            "hpo_worst_case_score": r["hpo_worst_case_score"],
            "alpha":                alpha,
        })
    comp_path = output_dir / "comparison_summary.csv"
    pd.DataFrame(rows).to_csv(comp_path, index=False)
    print(f"\nSaved comparison summary   → {comp_path}")

    # ── 2. Incumbent scores CSV (all methods × all incumbents) ────────────────
    # One row per incumbent_idx.  For each method: target score, source score,
    # and transfer_gap = (source − target) / |source|  (relative degradation,
    # positive = worse on target domain; dimensionless fraction).
    max_n = max(results[m]["n_incumbents"] for m in methods)
    score_rows = []
    for i in range(max_n):
        row: dict = {"incumbent_idx": i}
        for m in methods:
            t_scores = results[m]["incumbent_scores"]
            s_scores = results[m]["source_scores"]
            row[m]             = float(t_scores[i]) if i < len(t_scores) else float("nan")
            row[f"{m}_source"] = float(s_scores[i]) if i < len(s_scores) else float("nan")
            if i < len(t_scores) and not np.isnan(s_scores[i]) and s_scores[i] != 0.0:
                row[f"{m}_transfer_gap"] = float(
                    (s_scores[i] - t_scores[i]) / abs(s_scores[i])
                )
            else:
                row[f"{m}_transfer_gap"] = float("nan")
        score_rows.append(row)
    scores_path = output_dir / "incumbent_scores.csv"
    pd.DataFrame(score_rows).to_csv(scores_path, index=False)
    print(f"Saved incumbent scores     → {scores_path}")

    # ── 3. Per-domain comparison CSV (averaged over incumbents) ───────────────
    domain_rows = []
    for j, params in enumerate(target_domains):
        row2: dict = {"domain_idx": j}
        row2.update({f"param_{k}": v for k, v in params.items()})
        for m in methods:
            # all_rewards: (n_inc, n_targets, n_seeds)
            rw = results[m]["all_rewards"][:, j, :]   # (n_inc, n_seeds)
            row2[f"{m}_mean"] = float(rw.mean())
            row2[f"{m}_std"]  = float(rw.std())
        domain_rows.append(row2)
    domain_path = output_dir / "per_domain_comparison.csv"
    pd.DataFrame(domain_rows).to_csv(domain_path, index=False)
    print(f"Saved per-domain data      → {domain_path}")

    # ── 4. Target domains ─────────────────────────────────────────────────────
    td_path = output_dir / "target_domains.json"
    with open(td_path, "w") as f:
        json.dump(target_domains, f, indent=2)
    print(f"Saved target domains       → {td_path}")

    # ── 5. Per-method subdirectories ──────────────────────────────────────────
    for m in methods:
        method_dir = output_dir / m
        method_dir.mkdir(parents=True, exist_ok=True)

        incumbents  = results[m]["incumbents"]
        all_rewards = results[m]["all_rewards"]   # (n_inc, n_targets, n_seeds)
        inc_scores  = results[m]["incumbent_scores"]
        n_inc       = results[m]["n_incumbents"]

        # Per-method incumbent scores CSV (target + source)
        inc_score_rows = []
        for i in range(n_inc):
            src = float(results[m]["source_scores"][i])
            tgt = float(inc_scores[i])
            rel_gap = (src - tgt) / abs(src) if src != 0.0 and not np.isnan(src) else float("nan")
            inc_score_rows.append({
                "incumbent_idx": i,
                "score_S_i":     tgt,
                "source_score":  src,
                "transfer_gap":  rel_gap,
            })
        pd.DataFrame(inc_score_rows).to_csv(method_dir / "incumbent_scores.csv", index=False)

        # Per-incumbent subdirectory
        for i, (hp_config, rewards_i) in enumerate(zip(incumbents, all_rewards)):
            # rewards_i: (n_targets, n_seeds)
            inc_dir = method_dir / f"incumbent_{i}"
            inc_dir.mkdir(parents=True, exist_ok=True)

            # HP config YAML
            with open(inc_dir / "hp_config.yaml", "w") as f:
                yaml.dump(hp_config, f, default_flow_style=False)

            # Raw results CSV  (domain, seed, reward)
            raw_rows = []
            for d_idx, (params, rewards_d) in enumerate(zip(target_domains, rewards_i)):
                for s_idx, (seed, reward) in enumerate(zip(seed_list, rewards_d)):
                    r3 = {"domain_idx": d_idx, "seed": seed, "reward": float(reward)}
                    r3.update({f"param_{k}": v for k, v in params.items()})
                    raw_rows.append(r3)
            pd.DataFrame(raw_rows).to_csv(inc_dir / "results.csv", index=False)

            # Per-domain summary
            dom_rows = []
            for d_idx, (params, rewards_d) in enumerate(zip(target_domains, rewards_i)):
                r4 = {
                    "domain_idx": d_idx,
                    "mean": float(rewards_d.mean()),
                    "std":  float(rewards_d.std()),
                    "min":  float(rewards_d.min()),
                    "max":  float(rewards_d.max()),
                    "S_i":  float(inc_scores[i]),
                }
                r4.update({f"param_{k}": v for k, v in params.items()})
                dom_rows.append(r4)
            pd.DataFrame(dom_rows).to_csv(inc_dir / "per_domain_summary.csv", index=False)

        # Method-level summary JSON
        summary_out = {
            "metadata": {
                "method":    m,
                "env_name":  env_cfg.get("name"),
                "algorithm": algorithm,
                "n_incumbents": n_inc,
                "seeds":     seed_list,
                "alpha":     alpha,
            },
            "hpo_reliability": {
                "hpo_mean":             results[m]["hpo_mean"],
                "hpo_std":              results[m]["hpo_std"],
                "hpo_var":              results[m]["hpo_var"],
                "hpo_min":              results[m]["hpo_min"],
                "hpo_max":              results[m]["hpo_max"],
                "hpo_mean_var_score":   results[m]["hpo_mean_var_score"],
                "hpo_worst_case_score": results[m]["hpo_worst_case_score"],
            },
            "incumbent_scores": [float(s) for s in inc_scores],
        }
        with open(method_dir / "summary.json", "w") as f:
            json.dump(summary_out, f, indent=2, cls=_NumpyEncoder)

        print(f"  Saved {m:20s} → {method_dir}/")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare GPUCB / RAHBO / ERAHBO on held-out DR target domains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Environment & algorithm ───────────────────────────────────────────────
    p.add_argument(
        "--env-config",
        required=True,
        help="Environment config: YAML stem (e.g. 'cc_cartpole_dr') or absolute path.",
    )
    p.add_argument(
        "--algorithm",
        required=True,
        choices=["dqn", "ppo", "sac"],
        help="RL algorithm.",
    )

    # ── Method specification (three ways to specify) ──────────────────────────
    method_grp = p.add_argument_group("Method specification (use --methods OR the named flags)")
    method_grp.add_argument(
        "--methods",
        nargs="+",
        metavar="NAME:PATH",
        default=[],
        help=(
            "Space-separated list of 'NAME:path' pairs.  PATH may be either a "
            "run directory (containing runhistory.csv / incumbent.csv) or a YAML "
            "file holding an HP config directly.  Example: "
            "GPUCB:results/gpucb/run1 RAHBO:results/rahbo/run1"
        ),
    )
    method_grp.add_argument("--gpucb",  metavar="PATH", default=None, help="Run dir / YAML for GPUCB.")
    method_grp.add_argument("--rahbo",  metavar="PATH", default=None, help="Run dir / YAML for RAHBO.")
    method_grp.add_argument("--erahbo", metavar="PATH", default=None, help="Run dir / YAML for ERAHBO.")

    # ── Evaluation settings ───────────────────────────────────────────────────
    p.add_argument("--n-targets",           type=int,   default=10,   help="Number of target domains.")
    p.add_argument("--n-seeds",             type=int,   default=10,   help="Training seeds per domain.")
    p.add_argument("--seed-offset",         type=int,   default=0,    help="First seed value.")
    p.add_argument("--target-seed",         type=int,   default=0,    help="RNG seed for domain sampling.")
    p.add_argument("--n-total-timesteps",   type=float, default=None, help="Training budget override.")
    p.add_argument("--n-eval-steps",        type=int,   default=None, help="Eval steps override.")
    p.add_argument("--n-eval-episodes",     type=int,   default=None, help="Eval episodes override.")

    # ── Metric coefficients ────────────────────────────────────────────────────
    p.add_argument("--alpha", type=float, default=1.0,  help="Variance penalty: used in mean − alpha·Var for RAHBO/ERAHBO selection and the hpo_mean_var_score metric.")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output-dir",  default=None,   help="Override output directory.")
    p.add_argument("--no-save",     action="store_true", help="Skip saving results to disk.")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Collect method list ───────────────────────────────────────────────────
    methods: list[tuple[str, str]] = []

    # Named shortcut flags first
    if args.gpucb:
        methods.append(("gpucb", args.gpucb))
    if args.rahbo:
        methods.append(("rahbo", args.rahbo))
    if args.erahbo:
        methods.append(("erahbo", args.erahbo))

    # --methods NAME:PATH pairs
    for item in args.methods:
        if ":" not in item:
            parser.error(f"--methods entries must be 'NAME:PATH', got '{item}'")
        name, path = item.split(":", 1)
        methods.append((name.strip(), path.strip()))

    if not methods:
        parser.error(
            "Specify at least one method with --methods NAME:PATH or "
            "--gpucb / --rahbo / --erahbo flags."
        )

    print("=" * 72)
    print("  DR Comparison Pipeline  (HPO Reliability Evaluation)")
    print("=" * 72)
    print(f"  Methods                  : {[m for m, _ in methods]}")
    print(f"  Env config               : {args.env_config}")
    print(f"  Algorithm                : {args.algorithm}")
    print(f"  Target domains           : {args.n_targets}  (seed={args.target_seed})")
    print(f"  Seeds per domain         : {args.n_seeds}")
    print(f"  alpha (RAHBO/ERAHBO / MV score): {args.alpha}")
    print(f"  (Each source dir will produce one S_i score per independent BO run)")

    # ── 1. Environment config ─────────────────────────────────────────────────
    env_cfg = load_env_config(args.env_config)
    dr_config = env_cfg.get("domain_randomization", {})
    if not dr_config:
        print(
            f"\nWarning: no 'domain_randomization' section found in '{args.env_config}'. "
            "All target domains will use nominal environment parameters.",
            file=sys.stderr,
        )

    # ── 2. Shared target domains ──────────────────────────────────────────────
    print(f"\nSampling {args.n_targets} shared target domains (seed={args.target_seed}) ...")
    target_domains = sample_target_domains(dr_config, args.n_targets, rng_seed=args.target_seed)
    for i, td in enumerate(target_domains):
        params_str = ", ".join(f"{k}={v:.4f}" for k, v in td.items())
        print(f"  Domain {i:>3d}: {params_str}")

    # ── 3. Seed list ──────────────────────────────────────────────────────────
    seed_list = list(range(args.seed_offset, args.seed_offset + args.n_seeds))

    # ── 4. Output directory ───────────────────────────────────────────────────
    if not args.no_save:
        if args.output_dir:
            output_dir: Path | None = Path(args.output_dir)
        else:
            env_name_slug = env_cfg["name"].replace("-", "_").lower()
            methods_slug = "_vs_".join(m for m, _ in methods).lower()
            output_dir = Path("results") / "dr_compare" / f"{args.algorithm}_{env_name_slug}" / methods_slug
    else:
        output_dir = None

    # ── 5. Run all methods ────────────────────────────────────────────────────
    results = run_comparison(
        methods=methods,
        env_cfg=env_cfg,
        algorithm=args.algorithm,
        target_domains=target_domains,
        seed_list=seed_list,
        n_total_timesteps=args.n_total_timesteps,
        n_eval_steps=args.n_eval_steps,
        n_eval_episodes=args.n_eval_episodes,
        alpha=args.alpha,
        output_dir=output_dir,
    )

    # ── 6. Print comparison ───────────────────────────────────────────────────
    print_comparison(results, target_domains, alpha=args.alpha)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    if not args.no_save:
        assert output_dir is not None
        save_comparison(
            output_dir=output_dir,
            results=results,
            target_domains=target_domains,
            seed_list=seed_list,
            env_cfg=env_cfg,
            algorithm=args.algorithm,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
