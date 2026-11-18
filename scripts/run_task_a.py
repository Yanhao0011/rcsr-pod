#!/usr/bin/env python3
"""Task A — FARS frame-confound robustness of the RCSR-POD diagnostic.

Additive robustness run (does NOT change any reported number). For each credible
PO violator, output the four per-threshold PSC^(t) contributions the diagnostic
computes internally, the realizing threshold pair (argmax gap), whether it uses
the fatal cut t=3, and a non-fatal-ladder re-run (restrict the max to t<=2) with
its one-sided delta-method LCB. Reuses the EXACT design + Firth fitter from
build_gol_ordered_models so the full-ladder gap reproduces the reported values.

Usage:
    python run_task_a.py --frame outputs/kabco_person_level_clean.csv \
                         --out outputs/task_a_frame_confound.csv
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from po_diagnostic_core import firth_logit  # noqa: E402

# --- design pipeline replicated from build_gol_ordered_models (avoids importing
#     mord/statsmodels; MUST stay byte-identical to that script's encoding) -----
PATHWAY_FEATURES = {
    "infrastructure": ["RUR_URB", "REL_ROAD", "ROUTE", "FUNC_SYS", "TYP_INT",
                       "WRK_ZONE", "LGT_COND", "WEATHER", "HARM_EV", "MAN_COLL",
                       "mode_acc_type", "mode_impact1"],
    "pavement": ["mode_vpavetyp", "mode_vsurcond", "veh_vspd_lim", "veh_trav_sp",
                 "veh_vnum_lan", "mode_valign", "mode_vprofile"],
}
CONTROL_FEATURES = ["AGE", "SEX", "DRINKING", "veh_dr_drink", "vehicle_count"]

DELTA = 0.01
ALPHA_SIG = 0.05

# Human-readable labels (encoded column -> spec reference label)
LABELS = {
    "HARM_EV_12.0": "Harm event (12)", "MAN_COLL_6.0": "Manner of collision (6)",
    "MAN_COLL_2.0": "Manner of collision (2)", "vehicle_count": "Vehicle count",
    "MAN_COLL_1.0": "Manner of collision (1)", "mode_impact1_6.0": "Primary impact (6)",
    "SEX": "Sex", "RUR_URB_2.0": "Rural-urban (2)",
    "mode_vpavetyp_MISSING": "mode_vpavetyp_MISSING", "veh_trav_sp": "Travel speed",
    "veh_vnum_lan": "Lane count", "veh_vspd_lim": "Speed limit",
    "DRINKING": "Driver drinking", "AGE": "Age",
    "mode_impact1_12.0": "Primary impact (12)", "mode_acc_type_13.0": "Accident type (13)",
    "MAN_COLL_7.0": "Manner of collision (7)", "mode_vpavetyp_2.0": "Pavement type (2)",
    "mode_valign": "Alignment",
}


# --- design v2 (2026-09-06) ---------------------------------------------------
# HARM_EV_12.0 ("collision with a motor vehicle in transport") equals the sum of the
# manner-of-collision dummies for 99.7% of records (MAN_COLL is defined only when
# HARM_EV = 12), so the block's threshold heterogeneity was split arbitrarily between
# the two by the penalty. The nested-redundant dummy is removed from every design.
EXCLUDE = ["HARM_EV_12.0"]
# MISSING levels whose prevalence equals the CRSS share of the fused frame (0.5682):
# the field is not collected in CRSS, so the dummy is the data-source indicator. They
# stay in the design (they absorb the frame composition) but are reported as FRAME
# TERMS, not as substantive violators.
FRAME_SHARE = 0.5682
FRAME_TERMS = {"infrastructure": ["RUR_URB_MISSING", "FUNC_SYS_MISSING", "ROUTE_MISSING"],
               "pavement": ["mode_vpavetyp_MISSING"]}


def is_frame_term(name: str, prevalence: float) -> bool:
    return name.endswith("_MISSING") and abs(prevalence - FRAME_SHARE) < 0.002


def encode_features(frame: pd.DataFrame, features):
    cols = [c for c in features if c in frame.columns]
    enc = pd.get_dummies(frame[cols], drop_first=True, dummy_na=False).astype(float)
    enc = enc.drop(columns=[c for c in EXCLUDE if c in enc.columns])
    return enc, list(enc.columns)


def fit_threshold_firth(X: pd.DataFrame, y: np.ndarray):
    """{t: (beta, se)} incl. 'const'; identical to build_gol_ordered_models."""
    levels = sorted(np.unique(y))
    cols = list(X.columns)
    idx = ["const"] + cols
    Xmat = np.column_stack([np.ones(len(X)), X.values])
    out = {}
    for t in levels[:-1]:
        by = (y > t).astype(int)
        if by.sum() == 0 or by.sum() == len(by):
            continue
        beta, se, ok = firth_logit(Xmat, by)
        out[t] = (pd.Series(beta, index=idx), pd.Series(se, index=idx))
        print(f"    threshold t>{t}: fitted (converged={ok}), n1={int(by.sum())}", flush=True)
    return out


def scale_no_mean(values: np.ndarray) -> np.ndarray:
    """StandardScaler(with_mean=False): divide by population std, zero-std -> 1."""
    sd = values.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return values / sd


def psc_profile(firth_full: dict, X: pd.DataFrame, feats_of_interest):
    ts = sorted(firth_full)
    feats = list(X.columns)
    Xv = X.values
    z = stats.norm.ppf(1 - ALPHA_SIG)
    per_t = {}
    for t in ts:
        beta, se = firth_full[t]
        const = float(beta.get("const", 0.0))
        bvec = beta.reindex(feats).to_numpy(dtype=float)
        per_t[t] = (bvec, se, const + Xv @ bvec)

    def logistic(x):
        return np.clip(1.0 / (1.0 + np.exp(-x)), 1e-12, 1 - 1e-12)

    rows = []
    for f in feats_of_interest:
        if f not in feats:
            print(f"    WARNING: feature {f} not in design; skipped", flush=True)
            continue
        jc = feats.index(f)
        xj = Xv[:, jc]
        shift, dse = {}, {}
        for t in ts:
            bvec, se, eta = per_t[t]
            bj = bvec[jc]
            se_bj = float(se.get(f, np.nan))
            eta0 = eta - bj * xj
            p0 = logistic(eta0)
            p = logistic(eta)
            shift[t] = float(np.mean(p - p0))
            # dPSC/dbeta_j = mean[x_ij * lambda(eta_i)] at the FULL predictor
            # (see po_diagnostic_core.ame_shift_table); the stripped p0 form was
            # a mis-evaluation of the derivative.
            g = float(np.mean(xj * p * (1 - p)))
            dse[t] = abs(g) * se_bj if np.isfinite(se_bj) else np.nan
        finite = [t for t in ts if np.isfinite(shift[t])]

        def max_over(pairs):
            mg, arg, mlcb = 0.0, None, -np.inf
            for (s, t) in pairs:
                gp = abs(shift[s] - shift[t])
                seg = np.sqrt(dse[s] ** 2 + dse[t] ** 2)
                if gp > mg:
                    mg, arg = gp, (s, t)
                mlcb = max(mlcb, gp - z * seg)
            return mg, arg, mlcb

        pairs_full = [(s, t) for i, s in enumerate(finite) for t in finite[i + 1:]]
        mg, arg, full_lcb = max_over(pairs_full)
        involves_t3 = arg is not None and 3 in arg
        nf = [t for t in finite if t <= 2]
        pairs_nf = [(s, t) for i, s in enumerate(nf) for t in nf[i + 1:]]
        mg_nf, arg_nf, lcb_nf = max_over(pairs_nf) if pairs_nf else (0.0, None, -np.inf)
        survives = bool(np.isfinite(lcb_nf) and lcb_nf > DELTA)
        rows.append(dict(
            feature=f, label=LABELS.get(f, f),
            PSC0=shift.get(0, np.nan), PSC1=shift.get(1, np.nan),
            PSC2=shift.get(2, np.nan), PSC3=shift.get(3, np.nan),
            Delta_full=mg, realizing_pair=f"({arg[0]},{arg[1]})" if arg else "NA",
            involves_t3="Y" if involves_t3 else "N",
            full_lcb=full_lcb, Delta_nf=mg_nf,
            realizing_pair_nf=f"({arg_nf[0]},{arg_nf[1]})" if arg_nf else "NA",
            lcb_nf=lcb_nf, survives_nonfatal="Y" if survives else "N",
        ))
    return rows


def run_pathway(pathway: str, frame: pd.DataFrame):
    print(f"\n=== pathway: {pathway} ===", flush=True)
    features = PATHWAY_FEATURES[pathway] + CONTROL_FEATURES
    encoded, names = encode_features(frame, features)
    keep = frame.dropna(subset=["kabco_ordinal"]).index
    y = pd.to_numeric(frame["kabco_ordinal"], errors="coerce").astype("Int64").dropna().astype(int).to_numpy()
    encoded = encoded.loc[encoded.index.isin(keep)].reset_index(drop=True)
    assert len(encoded) == len(y), (len(encoded), len(y))
    X = scale_no_mean(encoded.values)
    X_df = pd.DataFrame(X, columns=names)
    print(f"    design: n={len(X_df)}, p={X_df.shape[1]}", flush=True)
    firth_full = fit_threshold_firth(X_df, y)
    print(f"    fitted thresholds: {sorted(firth_full)}", flush=True)
    # credible set = the ame_violates==True features (top-8); read from the frame's
    # own diagnostic if provided, else fall back to the known credible feature list.
    credible = CREDIBLE[pathway]
    rows = psc_profile(firth_full, X_df, credible)
    for r in rows:
        r["pathway"] = pathway
    return rows


CREDIBLE = {
    # FULL-FRAME credible sets (ame_violates==True, delta=0.01), CORRECTED gradient:
    # 10 infra + 10 pave (MAN_COLL_7.0 dropped: corrected LCB 0.0032 < delta).
    "infrastructure": ["HARM_EV_12.0", "MAN_COLL_6.0", "vehicle_count", "MAN_COLL_1.0",
                       "mode_impact1_12.0", "MAN_COLL_2.0", "mode_impact1_6.0", "SEX",
                       "mode_acc_type_13.0", "AGE"],
    "pavement": ["mode_vpavetyp_MISSING", "veh_trav_sp", "vehicle_count", "veh_vnum_lan",
                 "mode_vpavetyp_2.0", "SEX", "veh_vspd_lim", "AGE", "mode_valign", "DRINKING"],
}
# reported full-ladder gaps for the self-check (from table_s_rcpo_ame_sensitivity)
REPORTED = {"HARM_EV_12.0": 0.3391, "mode_vpavetyp_MISSING": 0.0958}  # full-frame expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="outputs/kabco_person_level_clean.csv")
    ap.add_argument("--out", default="outputs/task_a_frame_confound.csv")
    ap.add_argument("--pathways", default="infrastructure,pavement")
    args = ap.parse_args()

    frame = pd.read_csv(args.frame, low_memory=False)
    print(f"frame loaded: {len(frame)} rows", flush=True)
    all_rows = []
    for p in args.pathways.split(","):
        all_rows.extend(run_pathway(p.strip(), frame))

    df = pd.DataFrame(all_rows)
    cols = ["pathway", "label", "feature", "PSC0", "PSC1", "PSC2", "PSC3",
            "Delta_full", "realizing_pair", "involves_t3", "full_lcb",
            "Delta_nf", "realizing_pair_nf", "lcb_nf", "survives_nonfatal"]
    df = df[cols]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    # self-check: full-ladder gap reproduces the reported value
    print("\n=== SELF-CHECK (full Delta vs reported) ===")
    for f, rep in REPORTED.items():
        got = df.loc[df["feature"] == f, "Delta_full"]
        if len(got):
            g = float(got.iloc[0])
            print(f"  {f}: computed {g:.4f} vs reported {rep:.4f} "
                  f"{'OK' if abs(g - rep) < 0.01 else 'MISMATCH'}")

    # summary line
    print("\n=== TABLE A ===")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for p in df["pathway"].unique():
        sub = df[df["pathway"] == p]
        surv = int((sub["survives_nonfatal"] == "Y").sum())
        print(f"\n  {p}: {surv} of {len(sub)} credible violators survive on the non-fatal ladder t in {{0,1,2}}")
    fh = df[df["feature"] == "HARM_EV_12.0"]
    if len(fh):
        r = fh.iloc[0]
        print(f"\n  Flagship Harm-event(12) 0.40 gap realized at pair {r['realizing_pair']}, "
              f"involves t=3: {r['involves_t3']}, survives non-fatal: {r['survives_nonfatal']} "
              f"(Delta_nf={r['Delta_nf']:.4f}, LCB_nf={r['lcb_nf']:.4f})")


if __name__ == "__main__":
    main()
