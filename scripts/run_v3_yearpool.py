#!/usr/bin/env python3
"""Year-pooling check on the recoded v3 frame: auxiliary binary logit for serious-or-fatal injury (y >= 3) with nine
person, vehicle and roadway variables (age with unknown indicator, sex, person drinking, vehicle count, speed limit,
lane count with unknown indicator, light condition, weather, rural-urban), pooled versus year-specific coefficients.
Writes outputs/v3/yearpool_v3.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v3_frame import load_frame, OUT
fr = load_frame(); y = (fr.kabco_ordinal >= 3).astype(int).to_numpy()
num = fr[["AGE", "AGE_UNK", "vehicle_count", "veh_vspd_lim", "veh_vnum_lan", "veh_vnum_lan_UNK"]].astype(float)
cat = pd.get_dummies(fr[["SEX", "DRINKING", "LGT_COND", "WEATHER", "RUR_URB"]].astype(str), drop_first=True).astype(float)
X = pd.concat([num, cat], axis=1); X = (X - X.mean()) / X.std().replace(0, 1); X = sm.add_constant(X).to_numpy()
ll0 = sm.Logit(y, X).fit(disp=0, maxiter=200).llf; ll1 = 0.0; k = X.shape[1]
for yr in sorted(fr.YEAR.unique()):
    m = (fr.YEAR == yr).to_numpy(); ll1 += sm.Logit(y[m], X[m]).fit(disp=0, maxiter=200).llf
lr = 2 * (ll1 - ll0); df = k * (fr.YEAR.nunique() - 1)
res = dict(params_per_fit=int(k), lr=float(lr), df=int(df), p=float(stats.chi2.sf(lr, df)), per_record_ll_gain=float((ll1 - ll0) / len(y)), n=int(len(y)))
(OUT / "yearpool_v3.json").write_text(json.dumps(res, indent=1)); print(res)
