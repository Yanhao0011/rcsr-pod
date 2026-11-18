#!/usr/bin/env python3
"""LaTeX fragments for the appendix tables (Appendices A and C), generated from the result files (no hand transcription).
Writes v3_tab_sim.tex, v3_tab_app.tex and v3_tab_sens.tex; a missing input produces a placeholder fragment.
Inputs: RCSR_RESULTS (default outputs/v3; the shipped results are in results/v3), simulations from <RCSR_RESULTS>/sim29
(deployed bound: joint covariance, GLS pooling, cut selection), application with the selection controls (crash size,
person type), comparison without them from RCSR_NOCTRL (default results/v3/b29_nocontrol).
Output directory: RCSR_TABLES (default outputs/tables)."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V3 = Path(os.environ.get("RCSR_RESULTS", ROOT / "outputs" / "v3")); SIM = V3 / "sim29"
NOCTRL = Path(os.environ.get("RCSR_NOCTRL", ROOT / "results" / "v3" / "b29_nocontrol"))  # earlier run without the selection controls
OUT = Path(os.environ.get("RCSR_TABLES", ROOT / "outputs" / "tables")); OUT.mkdir(parents=True, exist_ok=True)
def F(v, d=3):
    """fixed decimals; '--' for missing; a true minus sign for negatives"""
    if pd.isna(v): return "--"
    s = f"{v:.{d}f}"
    if s.startswith("-"):
        s = s[1:] if float(s) == 0 else "$-$" + s[1:]
    return s
N = lambda n: f"{int(n):,}".replace(",", "{,}")
LAB = {"vehicle_count": "Vehicle count", "SEX": "Sex", "AGE": "Age", "DRINKING": "Drinking (person)", "veh_dr_drink": "Drinking driver",
       "veh_trav_sp": "Travel speed", "veh_vspd_lim": "Speed limit", "veh_vnum_lan": "Number of lanes", "mode_valign": "Alignment",
       "mode_vprofile": "Grade profile", "LGT_COND": "Light condition", "WEATHER": "Weather", "WRK_ZONE": "Work zone",
       "RUR_URB_MISSING": "Source indicator (CRSS)", "mode_vpavetyp_MISSING": "Source indicator (CRSS)",
       "persons_in_crash": "Crash size (log persons)", "person_type": "Person type",
       "mode_vpavetyp": "Pavement type", "mode_vsurcond": "Surface condition", "MAN_COLL": "Manner of collision", "mode_acc_type": "Accident type",
       "mode_impact1": "Initial impact point", "REL_ROAD": "Relation to roadway", "HARM_EV": "First harmful event", "RUR_URB": "Rural--urban",
       "TYP_INT": "Intersection type", "ROUTE": "Route signing", "FUNC_SYS": "Functional system"}
FRAME = {"RUR_URB_MISSING", "mode_vpavetyp_MISSING", "persons_in_crash"}


def lab(u):
    if u.endswith(" [group]"):
        return LAB.get(u.replace(" [group]", ""), u) + " (all levels)"
    if u in LAB: return LAB[u] + (" [frame]" if u in FRAME else "")
    for p in sorted([k for k in LAB if not k.endswith("MISSING")], key=len, reverse=True):
        if u.startswith(p + "_"):
            return f"{LAB[p]} ({u[len(p) + 1:].replace('.0', '').replace('_', ' ').replace('nonoccupant', 'non-occupant')})"
    return u.replace("_", r"\_")


def esc(s): return str(s).replace("_", r"\_")


def mcse(rates, R):
    """largest Monte Carlo standard error over the reported rates"""
    r = np.asarray([x for x in rates if pd.notna(x)], float)
    return float(np.sqrt(r * (1 - r) / R).max()) if len(r) else np.nan


def mcnote(rates, R):
    return f"Monte Carlo standard error of a rate $\\hat p$: $\\sqrt{{\\hat p(1-\\hat p)/R}}$ with $R = {R}$, at most {F(mcse(rates, R))} in this table."


def sci(x, d=0):
    """1e-12 -> $1 \times 10^{-12}$ (no raw Python e-notation in the tables)"""
    m, e = f"{x:.{d}e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


def write(name, body):
    (OUT / name).write_text(body, encoding="utf-8")


def table(caption, label, colspec, header, rows, note=None, size="\\small"):
    s = ["\\begin{table}[!ht]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size, "\\setlength{\\tabcolsep}{3.5pt}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
         header + " \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    if note: s += ["", "\\smallskip", f"\\parbox{{0.95\\textwidth}}{{\\footnotesize {note}}}"]
    return "\n".join(s + ["\\end{table}", ""])


def placeholder(label, what):
    return f"\\begin{{table}}[!ht]\\centering\\caption{{{what} (pending).}}\\label{{{label}}}\\end{{table}}\n"


def pow_tab():
    p = SIM / "sim_pow_summary.csv"
    if not p.exists(): return placeholder("stab:pow", "Boundary and power experiment")
    d = pd.read_csv(p)
    need = {"shape", "mult", "N", "Dtrue", "rate", "rate_marg", "brant", "cover"}
    if not need <= set(d.columns): return placeholder("stab:pow", "Boundary and power experiment (columns to be mapped)")
    c = d[d.ftype == "carrier"] if "ftype" in d else d
    c = c.sort_values(["shape", "mult", "N"])
    opt = lambda r, k, dd=3: F(getattr(r, k), dd) if hasattr(r, k) else "--"
    rows = [f"{ {'peak': 'peaked'}.get(r.shape, r.shape)} & {r.mult:g} & {F(r.Dtrue, 4)} & {N(r.N)} & {F(r.rate)} & {F(r.rate_marg)} & {opt(r, 'rate_nosel')} & {F(r.brant)} & {opt(r, 'bic')} & {F(r.cover)} & {opt(r, 'c_med', 2)} \\\\"
            for r in c.itertuples()]
    R = int(d.reps.iloc[0]) if "reps" in d else 300
    return table(f"Boundary and power experiment.",
                 "stab:pow", "lrrrrrrrrrr", "Profile & nominal & true $\\Delta$ & $N$ & Simult. & Marginal & No selection & Brant & BIC & Cover & $c_k$", rows,
                 "Nominal: target departure in units of $\\delta$; Simult.: share of binary covariates with LCB $> \\delta$; Marginal: $z$ bound over all cuts; No selection: max-$t$ bound over all cuts; Brant: joint-form $p < 0.05$; Cover: share of replications with LCB $\\le$ true $\\Delta$; $c_k$: median critical value.", "\\footnotesize")


def e1_tab():
    p = SIM / "sim_e1_summary.csv"
    if not p.exists(): return placeholder("stab:e1", "Sub-relevance departures")
    d = pd.read_csv(p).sort_values(["departure", "N"]); R = int(d.reps.iloc[0])
    rows = [f"{'exact PO' if r.departure == 0 else 'trivial ($0.04$)'} & {N(r.N)} & {int(r.n_cov)} & {F(r.max_Dtrue, 4)} & {F(r.brant_fp)} & {F(r.bic_fp)} & {F(r.rcsr_fp)} & {F(r.rcsr_marg_fp)} \\\\" for r in d.itertuples()]
    return table(f"Sub-relevance experiment.", "stab:e1", "lrrrrrrr",
                 "Design & $N$ & Covariates & max true $\\Delta$ & Brant & BIC & RCSR-POD & Marginal bound", rows,
                 "Brant: joint-form $p < 0.05$; BIC: joint Wald statistic $> \\mathrm{df}\\cdot\\log N$.")


def e2_tab():
    p = SIM / "sim_e2_summary.csv"
    if not p.exists(): return placeholder("stab:e2", "Detection versus true departure")
    d = pd.read_csv(p).sort_values(["N", "Dtrue"]); R = 200
    rows = [f"{N(r.N)} & {esc(r.feature)} & {F(r.Dtrue, 4)} & {F(r.brant_det)} & {F(r.bic_det)} & {F(r.rcsr_det)} & {F(r.rcsr_marg_det)} & {F(r.cover)} \\\\" for r in d.itertuples()]
    return table(f"Planted-gap experiment.", "stab:e2", "rlrrrrrr",
                 "$N$ & Covariate & true $\\Delta$ & Brant & BIC & RCSR-POD & Marginal & Cover", rows,
                 None, size="\\footnotesize")


def e4_tab():
    p = SIM / "sim_e4_summary.csv"
    if not p.exists(): return placeholder("stab:e4", "Many-covariate experiment")
    d = pd.read_csv(p).sort_values("N"); R = 200
    rows = [f"{N(r.N)} & {F(r.brant_raw_fwer)} & {F(r.brant_holm_fwer)} & {F(r.bic_fwer)} & {F(r.rcsr_lcb_fwer)} & {F(r.rcsr_holm_fwer)} & {F(r.rcsr_bh_fdp)} & {F(r.bic_power)} & {F(r.rcsr_lcb_power)} & {F(r.rcsr_holm_power)} & {F(r.rcsr_bh_power)} \\\\" for r in d.itertuples()]
    t1 = table(f"Many-covariate experiment: error rates and power.",
               "stab:e4", "rrrrrrrrrrr", r"$N$ & \multicolumn{2}{c}{Brant FWER} & BIC & \multicolumn{2}{c}{RCSR-POD FWER} & BH FDP & BIC & \multicolumn{3}{c}{RCSR-POD power} \\ & raw & Holm & FWER & spec.\ set & Holm & & power & spec.\ set & Holm & BH",
               rows, "Spec.\\ set: LCB $> \\delta$ without multiplicity adjustment.", size="\\footnotesize")
    specs = ["PO", "RCSR-POD", "oracle", "Brant", "BIC", "GOL"]
    rows = [f"{N(r.N)} & " + " & ".join(f"{F(getattr(r, 'rmse_' + s.replace('-', '_')) if hasattr(r, 'rmse_' + s.replace('-', '_')) else d.loc[r.Index, 'rmse_' + s], 4)}" for s in specs)
            + " & " + " & ".join(F(d.loc[r.Index, 'k_' + s], 1) for s in ["RCSR-POD", "Brant", "BIC"]) + f" & {F(100 * d.loc[r.Index, 'neg_GOL'], 1)} \\\\" for r in d.itertuples()]
    t2 = table(f"Many-covariate experiment: post-selection accuracy.",
               "stab:post", "rrrrrrrrrrr", r"$N$ & \multicolumn{6}{c}{RMSE} & \multicolumn{3}{c}{mean freed units} & GOL neg. \\ & PO & RCSR-POD & oracle & Brant & BIC & GOL & RCSR-POD & Brant & BIC & (\%)",
               rows, "Oracle: the eight violators freed; GOL: all 20 covariates freed; GOL neg.: share of test records with a non-positive fitted category probability.", size="\\footnotesize")
    return t1 + "\n" + t2


def inv_sep_frm_tab():
    out = []
    p = SIM / "sim_inv_summary.csv"
    if p.exists():
        d = pd.read_csv(p).sort_values(["feature", "N", "coding"]); R = 200
        rows = [f"{ {'categorical (group)': 'categorical [group]'}.get(r.feature, r.feature)} & {r.coding} & {N(r.N)} & {F(r.Dtrue, 4)} & {F(r.D_old, 4)} & {F(r.flag_old)} & {F(r.Dinv, 4)} & {F(r.lcb_inv, 4)} & {F(r.flag_inv)} \\\\" for r in d.itertuples()]
        meta = json.loads((SIM / "inv_meta.json").read_text()).get("group_refit_spread", {})
        note = "Old: replacement departure without re-profiling, not computed for the grouped categorical (--)."
        out.append(table(f"Coding-invariance experiment.",
                         "stab:inv", "llrrrrrrr", "Covariate & Coding & $N$ & true $\\Delta$ & Old $\\hat\\Delta$ & Old flag & Invariant $\\hat\\Delta$ & LCB & Inv.\\ flag", rows, note, size="\\footnotesize"))
    p = SIM / "sim_sep_summary.csv"
    if p.exists():
        d0 = pd.read_csv(p); R = 300
        d = d0[d0.feature == "separated indicator"].sort_values(["strength", "prev", "N"], ascending=[False, True, True])
        rows = [f"{r.strength} & {r.prev:.2f} & {N(r.N)} & {F(r.Dtrue, 4)} & {F(r.ml_separated)} & {F(r.top_events, 1)} & {F(r.rate_firth)} & {F(r.rate_ml)} & {F(r.brant_ml)} & {r.se_ratio:.0f} \\\\" for r in d.itertuples()]
        o = d0[d0.feature != "separated indicator"]
        note = "ML separated: share of replications in which the maximum-likelihood fatal-cut coefficient diverges; Top events: mean number of indicator records in the top category; SE ratio: median over replications of the ratio of the largest maximum-likelihood to the largest Firth standard error of the indicator."
        out.append(table(f"Separation experiment.", "stab:sep", "lrrrrrrrrr",
                         "Separation & Prev. & $N$ & true $\\Delta$ & ML separated & Top events & Firth flag & ML flag & Brant (ML) & SE ratio ML/Firth", rows, note, size="\\footnotesize"))
    p = SIM / "sim_frm_summary.csv"
    if p.exists():
        d = pd.read_csv(p).sort_values(["model", "feature", "N"], ascending=[False, True, True]); R = 200
        rows = [f"{'with' if r.model == 'with_source' else 'without'} & {esc(r.feature)} & {N(r.N)} & {F(r.Dtrue, 4)} & {F(r.Dmax, 4)} & {F(r.rate)} & {F(r.brant)} & {F(r.bic)} \\\\" for r in d.itertuples()]
        out.append(table(f"Frame-versus-mechanism experiment.",
                         "stab:frm", "llrrrrrr", "Source indicator & Covariate & $N$ & true $\\Delta$ & mean $\\hat\\Delta$ & RCSR-POD flag & Brant flag & BIC flag", rows,
                         None, size="\\footnotesize"))
    return "\n".join(out) if out else placeholder("stab:inv", "Invariance, separation and frame experiments")


def e5_tab():
    p = SIM / "sim_e5_summary.csv"
    if not p.exists(): return placeholder("stab:e5", "Extended design")
    d = pd.read_csv(p).sort_values(["utype", "unit", "N"]); R = 200
    rows = [f"{esc(r.unit)} & {r.utype.replace('_', ' ')} & {N(r.N)} & {F(r.Dtrue, 4)} & {F(r.rate)} & {F(r.rate_cl)} & {F(r.rate_marg)} & {F(r.cover)} & {F(r.cover_cl)} & {F(r.c_med, 2)} \\\\" for r in d.itertuples()]
    return table(f"Extended design.",
                 "stab:e5", "llrrrrrrrr", "Unit & Type & $N$ & true $\\Delta$ & Flag (model) & Flag (cluster) & Flag (marg.) & Cover (model) & Cover (cluster) & $c_k$", rows,
                 None, size="\\scriptsize")


def scale_tab():
    p = SIM / "sim_scale_summary.csv"
    if not p.exists(): return placeholder("stab:scale", "Application-scale design")
    d = pd.read_csv(p).sort_values(["N", "Dtrue"], ascending=[True, False]); R = int(d.reps.iloc[0])
    rows = [f"{N(r.N)} & {esc({'peak carrier': 'peaked covariate'}.get(r.unit, r.unit))} & {F(r.Dtrue, 4)} & {F(r.Dmax, 4)} & {F(r.rate)} & {F(r.rate_nosel)} & {F(r.rate_marg)} & {F(r.brant)} & {F(r.bic)} & {F(r.cover)} & {F(r.c_med, 2)} & {F(r.c_nosel_med, 2)} \\\\" for r in d.itertuples()]
    return table(f"Application-scale design.",
                 "stab:scale", "rlrrrrrrrrrr", "$N$ & Unit & true $\\Delta$ & mean $\\hat\\Delta$ & Flag & No sel. & Marg. & Brant & BIC & Cover & $c_k$ & $c_k$ (no sel.)", rows,
                 "Flag: bound with cut selection; No sel.: max-$t$ bound over all four cuts; Marg.: marginal bound.", size="\\footnotesize")


def primary_tabs():
    out = []
    for pw, title in [("infrastructure", "Infrastructure"), ("pavement", "Pavement")]:
        d = pd.read_csv(V3 / f"v3_primary_{pw}.csv")
        rows = [f"{lab(r.unit)} & {int(r.n_cols)} & {F(r.prevalence)} & {F(r.Dmax, 4)} & {int(r.Dmax_cut)} & {F(r.cl_all_lcb, 4)} & {F(r.cl_all_lcb_marg, 4)} & {F(r.m_all_lcb, 4)} & {F(r.Dmax_nf, 4)} & {F(r.cl_nf_lcb, 4)} & {F(r.rel_Dmax, 3)} & {F(r['cl_all_p0.01'], 3)} \\\\"
                for _, r in d.iterrows()]
        out.append(table(f"{title} set: departures and bounds by covariate.", f"stab:prim_{pw[:5]}",
                         "lrrrrrrrrrrr", "Covariate & Cols & Prev. & $\\Delta$ & Cut & LCB & LCB (marg.) & LCB (model) & $\\Delta_{\\text{nf}}$ & LCB$_{\\text{nf}}$ & $\\Delta/\\bar P$ & $p_k$", rows,
                         "Cols: number of coded columns; Prev.: share of records with a nonzero column; LCB: simultaneous cut-selected bound from the joint crash-cluster covariance; marg.: marginal bound; model: model-based covariance; nf: non-fatal cuts $t \\le 2$ only; $\\Delta/\\bar P$: largest cut-relative departure; $p_k$: bootstrap $p$-value for $\\Delta_k \\le 0.01$; --: not computed, because units with $\\Delta \\le \\delta/4$ are not bootstrapped.", "\\scriptsize"))
    return "\n".join(out)


def noctrl_tab():
    rows = []
    for pw, title in [("infrastructure", "Infrastructure"), ("pavement", "Pavement")]:
        a = pd.read_csv(NOCTRL / f"v3_primary_{pw}.csv").set_index("unit"); b = pd.read_csv(V3 / f"v3_primary_{pw}.csv").set_index("unit")
        units = list(dict.fromkeys(list(b.index) + list(a.index)))
        for u in units:
            ra = a.loc[u] if u in a.index else None; rb = b.loc[u] if u in b.index else None
            if max(ra.Dmax if ra is not None else 0, rb.Dmax if rb is not None else 0) < 0.005: continue
            g = lambda r, k, dd=4: F(r[k], dd) if r is not None else "--"
            cr = lambda r: ("--" if r is None else ("yes" if pd.notna(r.cl_all_lcb) and r.cl_all_lcb > 0.01 else "no"))
            rows.append(f"{title} & {lab(u)} & {g(ra, 'Dmax')} & {g(ra, 'cl_all_lcb')} & {cr(ra)} & {g(rb, 'Dmax')} & {g(rb, 'cl_all_lcb')} & {cr(rb)} \\\\")
    return table("Departures without and with the selection controls.", "stab:noctrl", "llrrlrrl",
                 r"Set & Unit & \multicolumn{3}{c}{Without controls} & \multicolumn{3}{c}{With controls (reported)} \\ & & $\Delta$ & LCB & Credible & $\Delta$ & LCB & Credible", rows,
                 "--: not in the model, or not bootstrapped ($\\Delta \\le \\delta/4$).", size="\\scriptsize")


def covwald_tab():
    rows = []
    for pw, title in [("infrastructure", "Infrastructure"), ("pavement", "Pavement")]:
        d = pd.read_csv(V3 / f"brant_v3_covariates_{pw}.csv").sort_values("chi2", ascending=False)
        rows += [f"{title} & {lab(r.unit)} & {int(r.df)} & {r.chi2:,.0f} & {'yes' if r.holm else 'no'} & {'yes' if r.bic else 'no'} \\\\".replace(",", "{,}") for r in d.itertuples()]
    return table("Grouped joint Wald test of PO by covariate.", "stab:covwald", "llrrll",
                 "Set & Covariate & df & $\\chi^2$ & Holm & BIC", rows, size="\\scriptsize")


def dummy_tabs():
    out = []
    for pw, title in [("infrastructure", "Infrastructure"), ("pavement", "Pavement")]:
        d = pd.read_csv(V3 / f"v3_{pw}.csv"); d = d[d.kind == "feature"].sort_values("Dmax", ascending=False).head(20)
        rows = [f"{lab(r.unit)} & {F(r.prevalence)} & {F(r.Dmax, 4)} & {int(r.Dmax_cut)} & {F(r.cl_all_lcb, 4)} & {F(r.m_all_lcb, 4)} & {F(r.j_all_lcb, 4)} \\\\" for _, r in d.iterrows()]
        out.append(table(f"{title} set: dummy-level departures.",
                         f"stab:dummy_{pw[:5]}", "lrrrrrr", "Feature & Prev. & $\\Delta$ & Cut & LCB (cluster) & LCB (model) & LCB (record sandwich)", rows, note="Numbers in parentheses are FARS/CRSS codes; record sandwich: joint sandwich covariance without crash clustering.", size="\\scriptsize"))
    return "\n".join(out)


def sens_tabs():
    out = []
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import run_v3_sensitivity as SV
    for mode, cap in [("strata", "Source-stratified diagnostic"),
                      ("weighted", "Design-weighted diagnostic")]:
        p = V3 / f"sens_{mode}.csv"
        if not p.exists():
            out.append(placeholder(f"stab:{mode}", cap)); continue
        d = pd.read_csv(p); rows = []
        main = {pw: set(json.loads((V3 / "v3_primary_summary.json").read_text())[pw]["credible"]) for pw in ["pavement", "infrastructure"]}
        for (an, pw), g in d.groupby(["analysis", "pathway"], sort=False):
            pr = SV.primary(g.reset_index(drop=True), pw).sort_values("Dmax", ascending=False)
            pr = pr[(pr.Dmax > 0.005) | pr.unit.isin(main[pw])]
            for _, r in pr.iterrows():
                rows.append(f"{pw.capitalize()} & {an.replace('strata_', '').upper() if 'strata' in an else 'weighted'} & {N(r.n)} & {lab(r.unit)} & {F(r.Dmax, 4)} & {int(r.Dmax_cut)} & {F(r.lcb, 4)} & {'yes' if r.lcb > 0.01 else 'no'} \\\\")
        out.append(table(cap + ".", f"stab:{mode}", "llrlrrrl",
                         "Set & Analysis & $N$ & Covariate & $\\Delta$ & Cut & LCB & Credible", rows,
                         "--: not bootstrapped ($\\Delta \\le \\delta/4$)." if any(" -- " in x for x in rows) else None, size="\\scriptsize"))
    p = V3 / "sens_refit.csv"
    if p.exists():
        d = pd.read_csv(p)
        rows = [f"{r.pathway.capitalize()} & {lab(r.unit)} & {'frame' if r.frame_term else ('yes' if r.credible else 'no')} & {F(r.Dmax, 4)} & {F(r.R, 4)} & {int(r.R_cut)} & {sci(r.max_abs_signed, 1)} \\\\" for r in d.itertuples()]
        out.append(table("Refit check.",
                         "stab:refit", "llcrrrr", "Set & Covariate & Credible & $\\Delta_k$ & $R_k$ & Cut & max $|$signed change$|$", rows, size="\\footnotesize"))
    else:
        out.append(placeholder("stab:refit", "Refit check"))
    rows = []
    rs = json.loads((V3 / "sens_refit_summary.json").read_text())
    for pw, title in [("infrastructure", "Infrastructure"), ("pavement", "Pavement")]:
        d = pd.read_csv(V3 / f"sens_poref_{pw}.csv")
        rows += [f"{title} & {lab(r.unit)} & {F(r.Dmax_gls, 4)} & {F(r.Dmax_poref, 4)} & {int(r.cut_poref)} \\\\" for r in d.itertuples() if max(r.Dmax_gls, r.Dmax_poref) > 0.005]
        rows.append(f"{title} & correlation over all units & \\multicolumn{{2}}{{c}}{{{F(rs[pw + '_poref']['corr'])}}} & \\\\")
    out.append(table("Departures under two pooled references.", "stab:poref", "llrrr",
                     "Set & Unit & $\\Delta$ (GLS) & $\\Delta$ (PO fit) & Cut (PO fit)", rows, size="\\scriptsize"))
    # Year pooling is reported in the text (Appendix B.3); no separate table here.
    return "\n".join(out)


def heldout_tab():
    d = pd.concat([pd.read_csv(V3 / f"heldout_v3_{p}.csv") for p in ["pavement", "infrastructure"]])
    w = d.pivot_table(index=["pathway", "fold"], columns="spec", values="logloss_trunc")
    specs = ["PO", "PO + frame terms", "RCSR-POD", "RCSR-POD (dummy units)", "BIC (covariate Wald)", "Marginal Brant", "GOL"]
    rows = [f"{pw.capitalize()} & {k} & " + " & ".join(F(w.loc[(pw, k), s], 4) for s in specs) + " \\\\" for pw, k in w.index]
    return table("Held-out log loss by fold.", "stab:heldout", "lrrrrrrrr",
                 "Set & Fold & PO & PO + frame terms & RCSR-POD & RCSR-POD (dummy) & BIC & Brant & GOL", rows, size="\\footnotesize")


def main():
    write("v3_tab_sim.tex", "\n".join([pow_tab(), e1_tab(), e2_tab(), e4_tab(), inv_sep_frm_tab(), e5_tab(), scale_tab()]))
    write("v3_tab_app.tex", "\n".join([primary_tabs(), covwald_tab(), noctrl_tab(), dummy_tabs()]))
    write("v3_tab_sens.tex", "\n".join([sens_tabs(), heldout_tab()]))
    print("written")


if __name__ == "__main__":
    main()
