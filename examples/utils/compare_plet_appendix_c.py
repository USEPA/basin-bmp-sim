#!/usr/bin/env python3
"""Compare model outputs against a raw PLET Appendix C implementation.

This script builds one scenario per BMP (CPS) where exactly one BMP is placed on
one selected parcel, then compares:
- model parcel loads (from this repository, mode=plet_rusle)
- independently encoded PLET equations (Appendix C formulas)

Notes
-----
- The script uses deterministic one-BMP scenarios:
  - n_scenarios = 1
  - bmp_limit_n = 1
  - parcel selection fixed to one PID
  - BMP selection fixed to one CPS
  - bmp_fail_rate = 0.0
- For wetland BMPs, an empty parcel_up mapping is used so treatment stays on the
  selected parcel only.
- The "raw PLET" computation uses realized initial/final parameters and
  concentrations from load_parameters/s1.parquet, then re-computes loads from
  Appendix C equations.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

# Make project imports work when running from examples/utils.
ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import load_and_validate_all
from src.logging_utils import make_logger
from src.model import Model

INCH_OVER_HA_TO_LITERS = 254_000.0
TON_PER_ACRE_TO_KG_PER_HA = 907.18474 / 0.40468564224
FAIL_TOL = 1.0e-9


@dataclass
class ScenarioComparison:
    cps: int
    pid: str
    pollutants: List[str]
    model_initial: Dict[str, float]
    model_final: Dict[str, float]
    raw_initial: Dict[str, float]
    raw_final: Dict[str, float]
    abs_diff_initial: Dict[str, float]
    abs_diff_final: Dict[str, float]
    pct_diff_initial: Dict[str, float]
    pct_diff_final: Dict[str, float]
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one-BMP plet_rusle model outputs against raw Appendix C equations."
    )
    parser.add_argument(
        "--base-config",
        default="examples/east_fork/east_fork_plet.yaml",
        help="Base YAML config to clone for one-BMP test scenarios.",
    )
    parser.add_argument(
        "--pid",
        default=None,
        help="Parcel ID to force in every scenario. Defaults to first PID from parcel_p input.",
    )
    parser.add_argument(
        "--out-dir",
        default="examples/east_fork/outputs_plet_appendix_c_compare",
        help="Directory for generated per-BMP scenario outputs and comparison tables.",
    )
    parser.add_argument(
        "--n-parcels",
        type=int,
        default=1,
        help="Number of parcels to test. Default=1 (single-parcel detailed output).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when --n-parcels > 1.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return {str(k).lower(): v for k, v in cfg.items()}


def _resolve_path_like(value: Any, base_dir: Path) -> Any:
    """Resolve filesystem-like values relative to ``base_dir`` when needed."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return value
    p = Path(text)
    if p.is_absolute():
        return str(p)

    # Support both config-dir-relative and repo-root-relative path styles.
    candidates = [
        (base_dir / p).resolve(),
        (ROOT / p).resolve(),
        (Path.cwd() / p).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Fall back to config-dir-relative for clearer downstream errors.
    return str((base_dir / p).resolve())


def _resolve_config_paths(cfg: Dict[str, Any], base_cfg_path: Path) -> Dict[str, Any]:
    """Return a copy of cfg with file paths resolved from the config file folder."""
    out = dict(cfg)
    base_dir = base_cfg_path.parent

    top_level_keys = [
        "domain",
        "parcels",
        "outlet_loc",
        "parcel_out",
        "parcel_up",
        "parcel_p",
        "pollutant_yield",
        "bmp_efficiency",
        "bmp_cost",
        "delivery_ratios",
        "outlet_target",
        "outlet_mean",
        "bmp_sel",
    ]
    for key in top_level_keys:
        if key in out:
            out[key] = _resolve_path_like(out.get(key), base_dir)

    lg = out.get("load_generation")
    if isinstance(lg, dict):
        lg2 = dict(lg)
        for key in (
            "plet_inputs",
            "rusle_inputs",
            "pollutant_concentrations",
            "groundwater_concentrations",
            "bmp_parameter_effects",
        ):
            if key in lg2:
                lg2[key] = _resolve_path_like(lg2.get(key), base_dir)
        out["load_generation"] = lg2

    return out


def _normalize_pid(value: Any) -> str:
    text = str(value).strip()
    if text == "":
        raise ValueError("PID cannot be empty")
    try:
        num = float(text)
    except ValueError:
        return text
    if np.isfinite(num) and num.is_integer():
        return str(int(num))
    return text


def _select_pid(base_cfg: Mapping[str, Any], explicit_pid: Optional[str]) -> str:
    if explicit_pid is not None:
        return _normalize_pid(explicit_pid)
    parcel_p = Path(str(base_cfg["parcel_p"]))
    if not parcel_p.is_absolute():
        parcel_p = parcel_p.resolve()
    df = pd.read_csv(parcel_p)
    if "pid" not in df.columns:
        raise ValueError(f"parcel_p file missing pid column: {parcel_p}")
    if df.empty:
        raise ValueError(f"parcel_p file has no rows: {parcel_p}")
    return _normalize_pid(df.iloc[0]["pid"])


def _pid_population(base_cfg: Mapping[str, Any]) -> List[str]:
    parcel_p = Path(str(base_cfg["parcel_p"]))
    if not parcel_p.is_absolute():
        parcel_p = parcel_p.resolve()
    df = pd.read_csv(parcel_p)
    if "pid" not in df.columns or df.empty:
        raise ValueError(f"parcel_p file must contain non-empty pid column: {parcel_p}")
    pids = [_normalize_pid(v) for v in df["pid"].tolist()]
    # Keep first occurrence order.
    return list(dict.fromkeys(pids))


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def _write_single_pid_probability(path: Path, pid: str) -> None:
    pd.DataFrame([{"pid": str(pid), "probability": 1.0}]).to_csv(path, index=False)


def _write_single_cps_selection(path: Path, cps: int, cps_values: Iterable[int]) -> None:
    rows = [
        {"cps": int(candidate), "probability": 1.0 if int(candidate) == int(cps) else 0.0}
        for candidate in cps_values
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_empty_parcel_up(path: Path) -> None:
    pd.DataFrame(columns=["pid", "pid_up"]).to_csv(path, index=False)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return default
    return float(s)


def _runoff_event_depth_in(
    annual_precip_in: float,
    rain_days: float,
    rain_correction_fraction: float,
    runoff_day_fraction: float,
    cn: float,
    ia_ratio: float,
) -> float:
    """Appendix C Eq. 2-5 implementation in event/depth form."""
    annual_precip_in = max(0.0, float(annual_precip_in))
    rain_days = max(0.0, float(rain_days))
    rain_correction_fraction = float(np.clip(rain_correction_fraction, 0.0, 1.0))
    runoff_day_fraction = float(np.clip(runoff_day_fraction, 0.0, 1.0))
    cn = float(np.clip(cn, 1.0e-6, 100.0))
    ia_ratio = max(0.0, float(ia_ratio))

    runoff_days = rain_days * runoff_day_fraction
    if runoff_days <= 0.0:
        return 0.0

    p_event = annual_precip_in * rain_correction_fraction / runoff_days
    s = max(0.0, (1000.0 / cn) - 10.0)
    ia = ia_ratio * s
    if p_event <= ia:
        return 0.0
    return (p_event - ia) ** 2 / max((p_event - ia + s), 1.0e-12)


def _annual_runoff_depth_in(params: Mapping[str, float]) -> float:
    q_event = _runoff_event_depth_in(
        annual_precip_in=params["annual_precip_in"],
        rain_days=params["rain_days"],
        rain_correction_fraction=params["rain_correction_fraction"],
        runoff_day_fraction=params["runoff_day_fraction"],
        cn=params["cn"],
        ia_ratio=params.get("ia_ratio", 0.0),
    )
    runoff_days = max(0.0, params["rain_days"] * params["runoff_day_fraction"])
    q_storm = q_event * runoff_days

    # Appendix C Eq. 6 irrigation extension.
    irr_depth = max(0.0, params.get("irrigation_depth_in", 0.0))
    irr_freq = max(0.0, params.get("irrigation_frequency", 0.0))
    irr_frac = float(np.clip(params.get("irrigated_fraction", 1.0), 0.0, 1.0))

    if irr_depth > 0.0 and irr_freq > 0.0 and irr_frac > 0.0:
        s = max(0.0, (1000.0 / max(params["cn"], 1.0e-6)) - 10.0)
        ia = max(0.0, params.get("ia_ratio", 0.0)) * s
        if irr_depth <= ia:
            q_irr_event = 0.0
        else:
            q_irr_event = (irr_depth - ia) ** 2 / max((irr_depth - ia + s), 1.0e-12)
        q_irr = q_irr_event * irr_freq * irr_frac
    else:
        q_irr = 0.0

    return (q_storm + q_irr) * max(0.0, params.get("runoff_multiplier", 1.0))


def _annual_infiltration_in(params: Mapping[str, float]) -> float:
    infil_frac = float(np.clip(params.get("infiltration_fraction", 0.0), 0.0, 1.0))
    rain_correct = float(np.clip(params.get("rain_correction_fraction", 1.0), 0.0, 1.0))
    p = max(0.0, params.get("annual_precip_in", 0.0))
    gw_multiplier = max(0.0, params.get("groundwater_multiplier", 1.0))
    return p * rain_correct * infil_frac * gw_multiplier


def _rusle_sediment_kg_ha(params: Mapping[str, float]) -> float:
    required = ("r", "k", "ls", "c", "p")
    if not all(k in params for k in required):
        return 0.0

    gross_ton_ac = 1.0
    for key in required:
        gross_ton_ac *= max(0.0, params[key])

    if "sdr" in params:
        sdr = float(np.clip(params["sdr"], 0.0, 1.0))
    else:
        sdr = 1.0

    sed_mult = max(0.0, params.get("sediment_multiplier", 1.0))
    deliv_mult = max(0.0, params.get("sediment_delivery_multiplier", 1.0))
    return gross_ton_ac * sdr * TON_PER_ACRE_TO_KG_PER_HA * sed_mult * deliv_mult


def _appendix_c_loads_kg_ha(
    params: Mapping[str, float],
    concentrations_mg_l: Mapping[str, float],
    gw_concentrations_mg_l: Mapping[str, float],
    pollutants: Sequence[str],
    *,
    groundwater_loads: bool,
) -> Dict[str, float]:
    """Independent per-pollutant annual load computation (kg/ha)."""
    q_runoff_in = _annual_runoff_depth_in(params)
    runoff_l_ha = q_runoff_in * INCH_OVER_HA_TO_LITERS
    infil_l_ha = _annual_infiltration_in(params) * INCH_OVER_HA_TO_LITERS
    sed_kg_ha = _rusle_sediment_kg_ha(params)
    enr = max(0.0, params.get("enrichment_ratio", 2.0))

    out: Dict[str, float] = {}
    for pollutant in pollutants:
        pol = str(pollutant).upper()
        c_runoff = max(0.0, float(concentrations_mg_l.get(pol, 0.0)))
        c_gw = max(0.0, float(gw_concentrations_mg_l.get(pol, 0.0)))

        runoff_load = c_runoff * runoff_l_ha / 1_000_000.0
        gw_load = (c_gw * infil_l_ha / 1_000_000.0) if (groundwater_loads and pol != "TSS") else 0.0

        if pol == "TSS":
            total = sed_kg_ha if sed_kg_ha > 0.0 else runoff_load
        elif pol == "TN":
            sed_frac = max(0.0, params.get("sediment_n_pct", 0.0)) / 100.0
            total = runoff_load + (sed_kg_ha * sed_frac * enr) + gw_load
        elif pol == "TP":
            sed_frac = max(0.0, params.get("sediment_p_pct", 0.0)) / 100.0
            total = runoff_load + (sed_kg_ha * sed_frac * enr) + gw_load
        else:
            total = runoff_load + gw_load

        total *= max(0.0, params.get(f"load_multiplier_{pol.lower()}", 1.0))
        out[pol] = float(max(0.0, total))

    return out


def _appendix_c_bmp_adjusted_final_kg_ha(
    initial_loads_kg_ha: Mapping[str, float],
    bmp_row: pd.Series,
    area_ha: float,
    pollutants: Sequence[str],
) -> Dict[str, float]:
    """Apply Appendix C-style BMP reduction using treated area and realized removal.

    For each pollutant, use:
    - effective treated-area share Aeff from treated load
    - realized efficiency e from removed/treated
    - final = initial * (1 - e * Aeff)
    """
    area_ha = max(float(area_ha), 1.0e-12)
    out: Dict[str, float] = {}

    for pollutant in pollutants:
        pol = str(pollutant).upper()
        initial = max(0.0, float(initial_loads_kg_ha.get(pol, 0.0)))
        treated_total = max(0.0, _as_float(bmp_row.get(f"treated_{pol}"), 0.0))
        removed_total = max(0.0, _as_float(bmp_row.get(f"removed_{pol}"), 0.0))

        if initial <= 0.0 or treated_total <= 0.0:
            out[pol] = float(initial)
            continue

        # Aeff is treated area fraction implied by treated mass.
        a_eff = np.clip(treated_total / max(initial * area_ha, 1.0e-12), 0.0, 1.0)
        # Realized efficiency in treated area.
        eff = np.clip(removed_total / max(treated_total, 1.0e-12), 0.0, 1.0)

        out[pol] = float(max(0.0, initial * (1.0 - (eff * a_eff))))

    return out


def _extract_param_sets(
    row: pd.Series,
    pollutants: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    parameter_keys = {
        "annual_precip_in",
        "rain_days",
        "rain_correction_fraction",
        "runoff_day_fraction",
        "cn",
        "ia_ratio",
        "r",
        "k",
        "ls",
        "c",
        "p",
        "sdr",
        "sediment_n_pct",
        "sediment_p_pct",
        "enrichment_ratio",
        "infiltration_fraction",
        "irrigated_fraction",
        "irrigation_depth_in",
        "irrigation_frequency",
        "runoff_multiplier",
        "groundwater_multiplier",
        "sediment_multiplier",
        "sediment_delivery_multiplier",
    }
    for pol in pollutants:
        parameter_keys.add(f"load_multiplier_{str(pol).lower()}")

    initial_params: Dict[str, float] = {}
    final_params: Dict[str, float] = {}
    for key in sorted(parameter_keys):
        ik = f"initial_{key}"
        fk = f"final_{key}"
        if ik in row and not pd.isna(row[ik]):
            initial_params[key] = _as_float(row[ik])
        if fk in row and not pd.isna(row[fk]):
            final_params[key] = _as_float(row[fk])

    init_conc: Dict[str, float] = {}
    final_conc: Dict[str, float] = {}
    init_gw_conc: Dict[str, float] = {}
    final_gw_conc: Dict[str, float] = {}

    for pol in pollutants:
        p = str(pol).lower()
        init_conc[str(pol).upper()] = _as_float(row.get(f"initial_concentration_{p}_mg_l", 0.0))
        final_conc[str(pol).upper()] = _as_float(row.get(f"final_concentration_{p}_mg_l", 0.0))
        init_gw_conc[str(pol).upper()] = _as_float(row.get(f"initial_groundwater_concentration_{p}_mg_l", 0.0))
        final_gw_conc[str(pol).upper()] = _as_float(row.get(f"final_groundwater_concentration_{p}_mg_l", 0.0))

    return initial_params, final_params, init_conc, final_conc, init_gw_conc, final_gw_conc


def _pct_diff(a: float, b: float) -> float:
    denom = max(abs(a), 1.0e-12)
    return abs(b - a) / denom * 100.0


def _run_one_cps(
    base_cfg: Dict[str, Any],
    cps: int,
    pid: str,
    root_out: Path,
) -> ScenarioComparison:
    scenario_out = root_out / f"pid_{_safe_token(pid)}" / f"cps_{int(cps)}"
    scenario_out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"cps_{int(cps)}_", dir=str(scenario_out)) as td:
        tdir = Path(td)
        parcel_p_single = tdir / "parcel_p_single.csv"
        bmp_sel_single = tdir / "bmp_sel_single.csv"
        parcel_up_empty = tdir / "parcel_up_empty.csv"

        _write_single_pid_probability(parcel_p_single, pid)
        _write_single_cps_selection(bmp_sel_single, int(cps), base_cfg.get("cps", []))
        _write_empty_parcel_up(parcel_up_empty)

        cfg = dict(base_cfg)
        cfg["mode"] = cfg.get("mode", "")  # harmless; preserves any accidental key casing behavior.
        cfg["n_scenarios"] = 1
        cfg["bmp_limit_n"] = 1
        cfg.pop("bmp_limit_usd", None)
        cfg["bmp_fail_rate"] = 0.0
        cfg["random_seed"] = 42
        cfg["verbose"] = False
        cfg["parallel"] = {"n_jobs": 1}
        cfg["outputs"] = str(scenario_out)
        cfg["parcel_p"] = str(parcel_p_single)
        cfg["parcel_up"] = str(parcel_up_empty)
        cfg["bmp_sel"] = str(bmp_sel_single)

        # Force an equations-only comparison baseline for the MODEL run:
        # always disable process-parameter effects regardless of YAML content.
        load_generation = cfg.get("load_generation")
        lg2 = dict(load_generation) if isinstance(load_generation, dict) else {}
        lg2["process_parameter_mode"] = False
        cfg["load_generation"] = lg2

        logger, _ = make_logger(scenario_out, verbose=False, console=False)
        data = load_and_validate_all(cfg, logger)

        if str(data.get("load_generation", {}).get("mode", "")).strip().lower() != "plet_rusle":
            raise ValueError("This comparison requires load_generation.mode = 'plet_rusle'.")

        model = Model(cfg, data, logger)
        model.run_all_scenarios()

        bmps_path = scenario_out / "bmps" / "s1.parquet"
        parcels_path = scenario_out / "parcels" / "s1.parquet"
        lp_path = scenario_out / "load_parameters" / "s1.parquet"

        if not bmps_path.exists() or not parcels_path.exists() or not lp_path.exists():
            raise RuntimeError(f"Expected output files not found for cps={cps} in {scenario_out}")

        bmps = pd.read_parquet(bmps_path)
        if bmps.empty:
            raise RuntimeError(f"No BMP records found for cps={cps}")
        if len(bmps) != 1:
            raise RuntimeError(f"Expected exactly one BMP record for cps={cps}, found {len(bmps)}")

        parcels = pd.read_parquet(parcels_path)
        lp = pd.read_parquet(lp_path)

        pid_str = str(pid)
        parcel_row = parcels[parcels["pid"].astype(str) == pid_str]
        load_row = lp[lp["pid"].astype(str) == pid_str]

        if parcel_row.empty:
            raise RuntimeError(f"Parcel pid={pid_str} not found in parcels output for cps={cps}")
        if load_row.empty:
            raise RuntimeError(f"Parcel pid={pid_str} not found in load_parameters output for cps={cps}")

        parcel_row = parcel_row.iloc[0]
        load_row = load_row.iloc[0]

        pollutants = [str(p).upper() for p in data["pollutants"]]

        initial_params, final_params, init_conc, final_conc, init_gw_conc, final_gw_conc = _extract_param_sets(
            load_row, pollutants
        )

        groundwater_loads = bool(data.get("load_generation", {}).get("groundwater_loads", False))

        raw_initial = _appendix_c_loads_kg_ha(
            initial_params,
            init_conc,
            init_gw_conc,
            pollutants,
            groundwater_loads=groundwater_loads,
        )

        process_mode_active = bool(data.get("load_generation", {}).get("process_parameter_mode", False))
        if process_mode_active:
            raw_final = _appendix_c_loads_kg_ha(
                final_params,
                final_conc,
                final_gw_conc,
                pollutants,
                groundwater_loads=groundwater_loads,
            )
        else:
            parcels_src = data.get("parcels")
            match = parcels_src[parcels_src["pid"].astype(str) == str(pid)]
            if match.empty:
                raise RuntimeError(f"Could not find parcel area for pid={pid}")
            area_ha = _as_float(match.iloc[0].get("area_ha"), 0.0)
            raw_final = _appendix_c_bmp_adjusted_final_kg_ha(
                raw_initial,
                bmps.iloc[0],
                area_ha,
                pollutants,
            )

        model_initial: Dict[str, float] = {}
        model_final: Dict[str, float] = {}
        for pol in pollutants:
            model_initial[pol] = _as_float(parcel_row.get(f"baseline_{pol}"))
            model_final[pol] = _as_float(parcel_row.get(f"final_{pol}"))

        abs_diff_initial = {pol: abs(raw_initial[pol] - model_initial[pol]) for pol in pollutants}
        abs_diff_final = {pol: abs(raw_final[pol] - model_final[pol]) for pol in pollutants}
        pct_diff_initial = {pol: _pct_diff(model_initial[pol], raw_initial[pol]) for pol in pollutants}
        pct_diff_final = {pol: _pct_diff(model_final[pol], raw_final[pol]) for pol in pollutants}

        return ScenarioComparison(
            cps=int(cps),
            pid=pid_str,
            pollutants=pollutants,
            model_initial=model_initial,
            model_final=model_final,
            raw_initial=raw_initial,
            raw_final=raw_final,
            abs_diff_initial=abs_diff_initial,
            abs_diff_final=abs_diff_final,
            pct_diff_initial=pct_diff_initial,
            pct_diff_final=pct_diff_final,
            output_dir=scenario_out,
        )


def _print_comparison(comp: ScenarioComparison) -> None:
    print(f"\n=== CPS {comp.cps} | PID {comp.pid} ===")
    print(f"output_dir: {comp.output_dir}")
    print("pollutant | model_baseline | raw_baseline | abs_diff | pct_diff | model_final | raw_final | abs_diff | pct_diff")
    for pol in comp.pollutants:
        print(
            f"{pol:9s} | "
            f"{comp.model_initial[pol]:13.6f} | {comp.raw_initial[pol]:12.6f} | "
            f"{comp.abs_diff_initial[pol]:8.6f} | {comp.pct_diff_initial[pol]:8.4f}% | "
            f"{comp.model_final[pol]:10.6f} | {comp.raw_final[pol]:9.6f} | "
            f"{comp.abs_diff_final[pol]:8.6f} | {comp.pct_diff_final[pol]:8.4f}%"
        )


def _print_caveats(process_mode_ignored: bool) -> None:
    print("\nImportant caveats")
    caveats = [
        "Only one BMP placement is forced per scenario (bmp_limit_n=1).",
        "Wetland BMPs can impact upstream parcels in full model runs; this script uses an empty parcel_up map so each scenario remains single-parcel treatment.",
        "If your base config enables extensions beyond canonical Appendix C (for example multipliers and groundwater options), those are included here to match your model context.",
        "Small differences can still occur from floating-point rounding and any model-side clipping/guards.",
    ]
    if process_mode_ignored:
        caveats.append(
            "The input YAML had load_generation.process_parameter_mode=true, but this script intentionally ignored it (forced false) to keep an apples-to-apples comparison against raw PLET equations."
        )
    for idx, text in enumerate(caveats, start=1):
        print(f"{idx}. {text}")


def _is_failed(comp: ScenarioComparison, pollutant: str, tol: float = FAIL_TOL) -> bool:
    return (comp.abs_diff_initial[pollutant] > tol) or (comp.abs_diff_final[pollutant] > tol)


def _print_multi_parcel_summary(comparisons: List[ScenarioComparison], sampled_pids: List[str]) -> None:
    failures: List[Tuple[str, int, str, float, float]] = []
    failed_pid_set = set()

    for comp in comparisons:
        for pol in comp.pollutants:
            if _is_failed(comp, pol):
                failed_pid_set.add(comp.pid)
                failures.append(
                    (
                        comp.pid,
                        comp.cps,
                        pol,
                        comp.abs_diff_initial[pol],
                        comp.abs_diff_final[pol],
                    )
                )

    print("\nBatch Comparison Summary")
    print(f"sampled_fields={len(sampled_pids)}")
    print(f"fields_failed={len(failed_pid_set)}")
    print(f"failed_cps_pollutant_tests={len(failures)}")

    if failures:
        print("failed_tests (pid, cps, pollutant, baseline_abs_diff, final_abs_diff):")
        for pid, cps, pol, bdiff, fdiff in failures:
            print(f"{pid}, {cps}, {pol}, {bdiff:.12g}, {fdiff:.12g}")
    else:
        print("No failed cps/pollutant tests.")


def main() -> None:
    args = parse_args()

    if args.n_parcels < 1:
        raise ValueError("--n-parcels must be >= 1")

    base_cfg_path = Path(args.base_config)
    if not base_cfg_path.is_absolute():
        # First honor caller's current working directory, then fall back to
        # repo-root-relative resolution for convenience.
        cwd_candidate = base_cfg_path.resolve()
        root_candidate = (ROOT / base_cfg_path).resolve()
        if cwd_candidate.exists():
            base_cfg_path = cwd_candidate
        else:
            base_cfg_path = root_candidate
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")

    base_cfg = _resolve_config_paths(_load_yaml(base_cfg_path), base_cfg_path)
    process_mode_in_yaml = bool(
        isinstance(base_cfg.get("load_generation"), dict)
        and base_cfg.get("load_generation", {}).get("process_parameter_mode", False)
    )

    if args.n_parcels == 1:
        pid = _select_pid(base_cfg, args.pid)
        sampled_pids = [pid]
    else:
        population = _pid_population(base_cfg)
        if not population:
            raise ValueError("No parcel IDs found for sampling")
        n = min(args.n_parcels, len(population))
        rng = np.random.default_rng(args.sample_seed)
        sampled_pids = [population[i] for i in rng.choice(len(population), size=n, replace=False)]
        if args.pid is not None:
            print("Note: --pid ignored because --n-parcels > 1 (random parcel sampling mode).")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cps_values = [int(x) for x in base_cfg.get("cps", [])]
    if not cps_values:
        raise ValueError("No cps values found in base config.")

    if args.n_parcels == 1:
        print(f"Running Appendix C comparison with pid={sampled_pids[0]} across cps={cps_values}")
    else:
        print(
            f"Running Appendix C comparison with n_parcels={len(sampled_pids)} "
            f"(seed={args.sample_seed}) across cps={cps_values}"
        )
    all_comparisons: List[ScenarioComparison] = []

    for pid in sampled_pids:
        for cps in cps_values:
            comp = _run_one_cps(base_cfg, cps=cps, pid=pid, root_out=out_dir)
            if args.n_parcels == 1:
                _print_comparison(comp)
            all_comparisons.append(comp)

    rows: List[Dict[str, Any]] = []
    for comp in all_comparisons:
        for pol in comp.pollutants:
            rows.append(
                {
                    "cps": comp.cps,
                    "pid": comp.pid,
                    "pollutant": pol,
                    "model_baseline_kg_ha": comp.model_initial[pol],
                    "raw_baseline_kg_ha": comp.raw_initial[pol],
                    "baseline_abs_diff": comp.abs_diff_initial[pol],
                    "baseline_pct_diff": comp.pct_diff_initial[pol],
                    "model_final_kg_ha": comp.model_final[pol],
                    "raw_final_kg_ha": comp.raw_final[pol],
                    "final_abs_diff": comp.abs_diff_final[pol],
                    "final_pct_diff": comp.pct_diff_final[pol],
                }
            )

    summary_path = out_dir / "appendix_c_comparison_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    manifest = {
        "base_config": str(base_cfg_path),
        "pids": sampled_pids,
        "n_parcels": len(sampled_pids),
        "sample_seed": int(args.sample_seed),
        "cps": cps_values,
        "summary_csv": str(summary_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote summary: {summary_path}")
    if args.n_parcels > 1:
        _print_multi_parcel_summary(all_comparisons, sampled_pids)
    _print_caveats(process_mode_ignored=process_mode_in_yaml)


if __name__ == "__main__":
    main()
