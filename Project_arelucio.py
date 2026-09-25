"""
This single script runs the following steps:
    1. data pipeline + three baselines (naive mean / Q1 carry / Q1 x ratio)
    2. XGBoost magnitude + sign models
    3. identity-based blending of the XGBoost predictions
    4. a GRU deep-learning model
    5. an XGBoost/GRU hybrid blend

The below five files are generated:
    submission_ratio_baseline.csv
    submission_xgb.csv
    submission_xgb_blend.csv
    submission_gru.csv
    submission_hybrid.csv
"""

import argparse
import itertools
import os
import pickle
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn

# ------------------------------------------------------------------ intermediate-file cache
# Every file that is NOT one of the final submission_*.csv deliverables is written under this
# folder instead of the current directory, and the folder is removed once the whole pipeline
# finishes successfully
CACHE_DIR = "_pipeline_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache(name):
    return os.path.join(CACHE_DIR, name)


"""
Step 1: data pipeline + baselines

What this does:
  1. Loads the train and the test files
  2. Cleans the 4 categorical columns
  3. Builds a grouped 10% holdout + 5 CV folds
  4. Scores three baselines with CV, using the grader's sMAPE:
       - naive mean       (rubric baseline)
       - Q1 carry-forward (predict last quarter's value)
       - Q1 x ratio       (Q1 value x median Q0/Q1 ratio learned on the training folds)
  5. Writes folds.csv, oof_ratio_baseline.csv, and submission_ratio_baseline.csv
"""

# ------------------------------------------------------------------ config
TRAIN_FILE = "train.csv"
TEST_FILE = "test_individual.csv"
SEED = 42
N_FOLDS = 5
HOLDOUT_FRAC = 0.10

TARGETS = [
    "Q0_TOTAL_ASSETS", "Q0_TOTAL_LIABILITIES", "Q0_TOTAL_STOCKHOLDERS_EQUITY",
    "Q0_GROSS_PROFIT", "Q0_COST_OF_REVENUES", "Q0_REVENUES",
    "Q0_OPERATING_INCOME", "Q0_OPERATING_EXPENSES", "Q0_EBITDA",
]
CAT_COLS = ["industry", "sector", "financialCurrency", "recommendationKey"]
# Value used for "missing" in each categorical column (kept as its own category)
MISSING_LABEL = {"industry": "Missing", "sector": "Unknown",
                 "financialCurrency": "Missing", "recommendationKey": "Missing"}
# Columns that identify the repeated synthetic record (591 copies)
GROUP_KEY = ["floatShares", "sharesOutstanding", "Q1_TOTAL_ASSETS", "Q1_REVENUES"]


# ------------------------------------------------------------------ metric
def smape(y, p):
    """Grader's sMAPE in %. A 0/0 term (y = p = 0) is counted as 0 error."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = (np.abs(y) + np.abs(p)) / 2
    err = np.where(denom == 0, 0.0, np.abs(y - p) / np.where(denom == 0, 1, denom))
    return 100 * err.mean()


def smape_table(y_df, p_df):
    """Per-target sMAPE plus the 9-target average (the grading metric)."""
    s = pd.Series({t: smape(y_df[t], p_df[t]) for t in TARGETS})
    s["AVERAGE"] = s[TARGETS].mean()
    return s


# ------------------------------------------------------------------ loading
def _read(path):
    df = pd.read_csv(path, low_memory=False)
    blank = df.isna().all(axis=1)
    print(f"  {os.path.basename(path)}: {len(df):,} rows, {blank.sum():,} empty rows dropped")
    df = df.loc[~blank].reset_index(drop=True).copy()
    df["Id"] = df["Id"].astype("int64")  # padding rows had made Id a float
    return df


def clean_categoricals(train, test=None):
    frames = [train] if test is None else [train, test]
    for df in frames:
        df["flag_profile_missing"] = (df["financialCurrency"] == "Missing").astype(int)  # 1,232 rows: all 4 missing
        df["flag_industry_missing"] = (df["industry"] == "Missing").astype(int)          # 1,344 rows

    for c in CAT_COLS:
        train[c] = train[c].fillna(MISSING_LABEL[c])
        cats = sorted(train[c].unique())
        train[c] = pd.Categorical(train[c], categories=cats)
        if test is not None:
            test[c] = test[c].fillna(MISSING_LABEL[c])
            unseen = sorted(set(test[c]) - set(cats))
            if unseen:
                print(f"  {c}: unseen test categories {unseen} -> '{MISSING_LABEL[c]}'")
                test.loc[test[c].isin(unseen), c] = MISSING_LABEL[c]
            test[c] = pd.Categorical(test[c], categories=cats)
    return train, test


def load_data():
    print("Loading:")
    train = _read(TRAIN_FILE)
    test = _read(TEST_FILE)

    assert train["Id"].is_unique, "Duplicate Ids in training data"
    features = [c for c in train.columns if c not in TARGETS + ["Id"]]
    if test is not None:
        missing = set(features) - set(test.columns)
        assert not missing, f"Test file missing predictors: {sorted(missing)}"
    train, test = clean_categoricals(train, test)
    features = [c for c in train.columns if c not in TARGETS + ["Id"]]  # now includes the 2 flags
    print(f"Train: {train.shape}, Test: {None if test is None else test.shape}, "
          f"predictors: {len(features)}")
    return train, test, features


# ------------------------------------------------------------------ folds
def make_folds(train, seed=SEED, n_folds=N_FOLDS, holdout_frac=HOLDOUT_FRAC):
    """
    Returns a Series: -1 = final holdout (checked once at the end), 0..n_folds-1 = CV fold.
    Rows sharing GROUP_KEY (the 591-copy record) always land in the same fold.
    """
    group = train.groupby(GROUP_KEY, sort=False).ngroup()
    rng = np.random.default_rng(seed)
    groups = rng.permutation(group.unique())
    sizes = group.value_counts()

    # holdout: take shuffled groups until ~holdout_frac of rows, skipping the big group
    n_hold, hold = holdout_frac * len(train), set()
    count = 0
    for g in groups:
        if count >= n_hold:
            break
        if sizes[g] > 1:
            continue
        hold.add(g)
        count += 1

    # CV folds: assign remaining groups to the currently smallest fold (balances sizes)
    fold_of, fold_rows = {}, np.zeros(n_folds)
    rest = [g for g in groups if g not in hold]
    for g in rest:
        k = int(np.argmin(fold_rows))
        fold_of[g] = k
        fold_rows[k] += sizes[g]
    fold = group.map(lambda g: -1 if g in hold else fold_of[g])
    return fold.astype(int)


# ------------------------------------------------------------------ baselines
def _q1(df, t):
    return df[t.replace("Q0_", "Q1_")].to_numpy(float)


def fit_ratio(train_part, t):
    """Median Q0/Q1 ratio over rows with a usable Q1, plus a fallback value."""
    y, q1 = train_part[t].to_numpy(float), _q1(train_part, t)
    ok = (q1 != 0) & np.isfinite(y / np.where(q1 == 0, 1, q1))
    return {"ratio": float(np.median(y[ok] / q1[ok])), "fallback": float(np.median(y))}


def predict_ratio(df, t, params):
    q1 = _q1(df, t)
    p = q1 * params["ratio"]
    p = np.where(q1 == 0, params["fallback"], p)  # never predict exactly 0 (scores 200%)
    return p


def run_cv_baselines(train, fold):
    cv_idx = fold >= 0
    oof = {name: pd.DataFrame(index=train.index[cv_idx], columns=TARGETS, dtype=float)
           for name in ["mean", "carry", "ratio"]}
    per_fold = []
    for k in range(N_FOLDS):
        tr, va = (fold >= 0) & (fold != k), fold == k
        for t in TARGETS:
            oof["mean"].loc[va[va].index, t] = train.loc[tr, t].mean()
            oof["carry"].loc[va[va].index, t] = _q1(train.loc[va], t)
            oof["ratio"].loc[va[va].index, t] = predict_ratio(train.loc[va], t, fit_ratio(train.loc[tr], t))
        per_fold.append(smape_table(train.loc[va, TARGETS], oof["ratio"].loc[va[va].index])["AVERAGE"])

    y = train.loc[cv_idx, TARGETS]
    res = pd.DataFrame({name: smape_table(y, oof[name]) for name in oof}).round(2)
    res.columns = ["Naive mean", "Q1 carry", "Q1 x ratio"]
    return res, oof, per_fold


# ------------------------------------------------------------------ main
def run_step1():
    train, test, features = load_data()

    fold = make_folds(train)
    print("\nFold sizes (-1 = holdout):", fold.value_counts().sort_index().to_dict())
    gid = train.groupby(GROUP_KEY, sort=False).ngroup()
    big = gid.value_counts().idxmax()
    print(f"Largest repeated-record group: {(gid == big).sum()} rows, "
          f"fold(s) {sorted(int(k) for k in fold[gid == big].unique())}")
    pd.DataFrame({"Id": train["Id"], "fold": fold}).to_csv(_cache("folds.csv"), index=False)

    print("\nCV sMAPE (%) on the 90% CV portion:")
    res, oof, per_fold = run_cv_baselines(train, fold)
    print(res.to_string())
    base = res.loc["AVERAGE", "Naive mean"]
    for col in ["Q1 carry", "Q1 x ratio"]:
        print(f"  {col}: {100 * (1 - res.loc['AVERAGE', col] / base):.1f}% reduction vs naive mean")
    print(f"  Q1 x ratio per-fold average sMAPE: {np.round(per_fold, 2)} (sd {np.std(per_fold):.2f})")
    oof["ratio"].assign(Id=train.loc[oof["ratio"].index, "Id"]).to_csv(_cache("oof_ratio_baseline.csv"), index=False)

    # Final ratio fit on all training rows (CV + holdout) -> safety-net submission
    if test is not None:
        sub = pd.DataFrame({"Id": test["Id"]})
        for t in TARGETS:
            sub[t] = predict_ratio(test, t, fit_ratio(train, t))
        assert list(sub.columns) == ["Id"] + TARGETS and len(sub) == len(test)
        assert sub[TARGETS].notna().all().all()
        sub.to_csv("submission_ratio_baseline.csv", index=False)
        print(f"\nWrote submission_ratio_baseline.csv ({len(sub):,} rows)")
    print("Wrote folds.csv, oof_ratio_baseline.csv (both under the intermediate cache)")


"""
Step 2: gradient boosting with XGBoost

Approach (per target, 9 targets):
  * Magnitude model: predicts  log1p|Q0| - log1p|Q1|  with a pseudo-Huber loss (L1-like).
    When the sign is right, each sMAPE term is 2*tanh(|log(pred/true)|/2), so an L1-like loss on
    log magnitude is closely aligned with the grading metric.
  * Sign model: classifier for P(Q0 > 0). A wrong sign scores exactly 200%, so the
    predicted sign is whichever is more probable.
  * Prediction = sign * max(magnitude, 1)  (never exactly 0: that scores 200%).
Raw numeric columns are never modified; engineered features are added alongside them.

Outputs:
  cv_results_xgb.csv          per-target CV sMAPE, sign accuracy, majority-sign accuracy
  oof_xgb.csv                 out-of-fold predictions (+ sign probabilities) for blending
  feature_importance_xgb.csv  gain importance per target (magnitude models)
  submission_xgb.csv          test predictions: Id + 9 targets
  holdout_xgb.csv             only with --holdout
"""

# ------------------------------------------------------------------ settings
# tree_method="hist" + enable_categorical=True lets XGBoost split pandas category columns natively
XGB_COMMON = dict(tree_method="hist", enable_categorical=True, max_cat_to_onehot=10,
                  learning_rate=0.1, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                  max_bin=128, random_state=SEED, n_jobs=-1)
HUBER_SLOPE = 0.2
XGB_REG = dict(XGB_COMMON, objective="reg:pseudohubererror", huber_slope=HUBER_SLOPE,
               eval_metric="mae", max_depth=7, min_child_weight=40, n_estimators=2000)
XGB_CLF = dict(XGB_COMMON, objective="binary:logistic", eval_metric="logloss",
               max_depth=6, min_child_weight=5, n_estimators=2000)
EARLY_STOP = 100        # rounds without improvement on the inner early-stopping split
ES_FRAC = 0.10          # share of each training fold held out for early stopping
FINAL_ITER_MULT = 1.10  # final model (more data) uses 10% more rounds than the CV average
SIGN_MODEL_MIN_COUNT = 30  # train a sign classifier whenever the rarer sign has >= 30 rows
                           # (revenues and operating expenses are never negative -> constant sign)

FIELDS = ["TOTAL_ASSETS", "TOTAL_CURRENT_ASSETS", "TOTAL_NONCURRENT_ASSETS",
          "TOTAL_LIABILITIES", "TOTAL_CURRENT_LIABILITIES", "TOTAL_NONCURRENT_LIABILITIES",
          "TOTAL_LIABILITIES_AND_EQUITY", "TOTAL_STOCKHOLDERS_EQUITY", "NET_INCOME",
          "GROSS_PROFIT", "COST_OF_REVENUES", "REVENUES", "OPERATING_INCOME",
          "OPERATING_EXPENSES", "EBITDA", "DEPRECIATION_AND_AMORTIZATION"]
BALANCE_SHEET = FIELDS[:8]          # these look altered (~ x -2) in Q3 and Q7
INCOME = FIELDS[8:]
SIGNED = ["OPERATING_INCOME", "EBITDA", "NET_INCOME", "GROSS_PROFIT", "TOTAL_STOCKHOLDERS_EQUITY"]


# ------------------------------------------------------------------ features
def slog(x):
    """Signed log: sign(x) * log(1 + |x|). Keeps sign, compresses scale."""
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))


def safe_div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.where(b != 0, a / np.where(b == 0, 1, b), np.nan)
    return np.clip(out, -1e3, 1e3)


def engineer(df):
    """Adds new columns only. Raw columns are passed through unchanged."""
    f = {}
    q = lambda k, name: df[f"Q{k}_{name}"].to_numpy(float)

    for name in FIELDS:
        f[f"fe_slog_Q1_{name}"] = slog(q(1, name))
        f[f"fe_g12_{name}"] = slog(q(1, name)) - slog(q(2, name))     # quarter-on-quarter
        f[f"fe_g15_{name}"] = slog(q(1, name)) - slog(q(5, name))     # year-on-year
        f[f"fe_g14_{name}"] = slog(q(1, name)) - slog(q(4, name))
        # Q3/Q7 skipped for balance-sheet fields because those two quarters are altered
        ks = [1, 2, 4, 5, 6, 8, 9, 10] if name in BALANCE_SHEET else range(1, 11)
        hist = np.column_stack([slog(q(k, name)) for k in ks])
        f[f"fe_mean_{name}"] = hist.mean(axis=1)
        f[f"fe_std_{name}"] = hist.std(axis=1)

    for name in SIGNED:
        ks = [1, 2, 4, 5, 6, 8, 9, 10] if name in BALANCE_SHEET else range(1, 11)
        vals = np.column_stack([q(k, name) for k in ks])
        f[f"fe_negshare_{name}"] = (vals < 0).mean(axis=1)
        f[f"fe_neg_Q1_{name}"] = (q(1, name) < 0).astype(int)
        f[f"fe_neg_Q2_{name}"] = (q(2, name) < 0).astype(int)
        f[f"fe_signflips_{name}"] = (np.diff(np.sign(vals), axis=1) != 0).sum(axis=1)

    rev = q(1, "REVENUES")
    for name in ["GROSS_PROFIT", "COST_OF_REVENUES", "OPERATING_EXPENSES", "OPERATING_INCOME",
                 "EBITDA", "NET_INCOME", "DEPRECIATION_AND_AMORTIZATION"]:
        f[f"fe_Q1_{name}_to_rev"] = safe_div(q(1, name), rev)
    ta = q(1, "TOTAL_ASSETS")
    f["fe_Q1_liab_to_assets"] = safe_div(q(1, "TOTAL_LIABILITIES"), ta)
    f["fe_Q1_equity_to_assets"] = safe_div(q(1, "TOTAL_STOCKHOLDERS_EQUITY"), ta)
    f["fe_Q1_curassets_to_assets"] = safe_div(q(1, "TOTAL_CURRENT_ASSETS"), ta)
    f["fe_Q1_curliab_to_liab"] = safe_div(q(1, "TOTAL_CURRENT_LIABILITIES"), q(1, "TOTAL_LIABILITIES"))

    # Versions of the Q0 identities, built from Q1 values
    f["fe_Q1_gp_minus_opex"] = slog(q(1, "GROSS_PROFIT") - q(1, "OPERATING_EXPENSES"))
    f["fe_Q1_oi_plus_da"] = slog(q(1, "OPERATING_INCOME") + q(1, "DEPRECIATION_AND_AMORTIZATION"))
    f["fe_Q1_assets_minus_liab"] = slog(ta - q(1, "TOTAL_LIABILITIES"))
    f["fe_Q1_rev_minus_cogs"] = slog(rev - q(1, "COST_OF_REVENUES"))

    # Trailing-12-month sums (income items are not altered in Q3)
    for name in ["REVENUES", "OPERATING_INCOME", "EBITDA", "GROSS_PROFIT", "OPERATING_EXPENSES"]:
        f[f"fe_ttm_{name}"] = slog(sum(q(k, name) for k in range(1, 5)))
    f["fe_ttm_rev_to_totalRevenue"] = safe_div(sum(q(k, "REVENUES") for k in range(1, 5)),
                                               df["totalRevenue"].to_numpy(float))

    # Flags for numeric placeholder values (the values themselves stay untouched)
    for c in ["auditRisk", "boardRisk", "compensationRisk", "shareHolderRightsRisk", "overallRisk"]:
        if c in df:
            f[f"fe_fill_{c}"] = (df[c] == 6.0).astype(int)
    f["fe_fill_targetprices"] = ((df["targetHighPrice"] == 30) & (df["targetLowPrice"] == 17)).astype(int)
    f["fe_fill_analysts"] = (df["numberOfAnalystOpinions"] == 6).astype(int)
    f["fe_fill_forwardEps"] = (df["forwardEps"] == 0.32).astype(int)

    return pd.concat([df, pd.DataFrame(f, index=df.index)], axis=1)


# ------------------------------------------------------------------ model wrappers
class GBM:
    """Thin wrapper around XGBRegressor / XGBClassifier."""

    def __init__(self, task, cat_cols):
        self.task, self.cat_cols, self.best_iter, self.model = task, cat_cols, None, None

    def fit(self, X, y, X_es=None, y_es=None, n_iter=None):
        params = dict(XGB_REG if self.task == "reg" else XGB_CLF)
        if self.task == "reg":
            params["base_score"] = float(np.median(y))
        if n_iter is not None:
            params["n_estimators"] = n_iter
        if X_es is not None:
            params["early_stopping_rounds"] = EARLY_STOP
        Model = xgb.XGBRegressor if self.task == "reg" else xgb.XGBClassifier
        self.model = Model(**params)
        if X_es is not None:
            self.model.fit(X, y, eval_set=[(X_es, y_es)], verbose=False)
            self.best_iter = self.model.best_iteration + 1   # best_iteration is 0-based
        else:
            self.model.fit(X, y, verbose=False)
            self.best_iter = params["n_estimators"]
        return self

    def predict(self, X):
        # with early stopping, the sklearn API predicts using the best iteration
        if self.task == "reg":
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]

    def importance(self, cols):
        gain = self.model.get_booster().get_score(importance_type="gain")
        return pd.Series(gain, dtype=float).reindex(cols).fillna(0.0)


# ------------------------------------------------------------------ targets
def magnitude_target(df, t):
    q1 = df[t.replace("Q0_", "Q1_")].to_numpy(float)
    return np.log1p(np.abs(df[t].to_numpy(float))) - np.log1p(np.abs(q1))


def combine_xgb(df, t, mag_pred, p_pos):
    q1 = df[t.replace("Q0_", "Q1_")].to_numpy(float)
    magnitude = np.expm1(mag_pred + np.log1p(np.abs(q1)))
    sign = np.where(p_pos >= 0.5, 1.0, -1.0)
    return sign * np.maximum(magnitude, 1.0)


class ConstantSign:
    """Stand-in for the sign classifier when one sign is overwhelmingly common."""

    def __init__(self, p_pos):
        self.p, self.best_iter = float(p_pos), 0

    def predict(self, X):
        return np.full(len(X), self.p)


def fit_pair(X_tr, df_tr, t, cat_cols, rng, n_iter=None):
    """Fits the magnitude + sign models for one target. Uses an inner split for early stopping.
    A sign classifier is trained whenever the rarer sign has at least SIGN_MODEL_MIN_COUNT rows
    (even 1-3% negatives are worth modelling: every wrong sign costs 200%). Otherwise, e.g. for
    revenues, which are never negative, the majority sign is used."""
    y_mag = magnitude_target(df_tr, t)
    y_pos = (df_tr[t].to_numpy(float) > 0).astype(int)
    need_clf = min(y_pos.sum(), len(y_pos) - y_pos.sum()) >= SIGN_MODEL_MIN_COUNT
    if n_iter is None:
        es = rng.random(len(X_tr)) < ES_FRAC
        need_clf = need_clf and len(np.unique(y_pos[es])) == 2 and len(np.unique(y_pos[~es])) == 2
        reg = GBM("reg", cat_cols).fit(X_tr[~es], y_mag[~es], X_tr[es], y_mag[es])
        clf = (GBM("clf", cat_cols).fit(X_tr[~es], y_pos[~es], X_tr[es], y_pos[es])
               if need_clf else None)
    else:
        reg = GBM("reg", cat_cols).fit(X_tr, y_mag, n_iter=n_iter[0])
        need_clf = need_clf and n_iter[1] >= 1   # 0 rounds = CV used the constant sign
        clf = GBM("clf", cat_cols).fit(X_tr, y_pos, n_iter=n_iter[1]) if need_clf else None
    if clf is None:
        clf = ConstantSign(1.0 if y_pos.mean() >= 0.5 else 0.0)
    return reg, clf


# ------------------------------------------------------------------ main
def run_step2(args):
    if args.gpu:
        XGB_REG["device"] = XGB_CLF["device"] = "cuda"
        print("Training on GPU (device='cuda')")
    print(f"XGBoost {xgb.__version__}, CPU cores available: {os.cpu_count()}")
    if int(xgb.__version__.split(".")[0]) < 2:
        raise SystemExit("Please upgrade: pip install -U 'xgboost>=2.0'")

    train, test, _ = load_data()
    folds = pd.read_csv(_cache("folds.csv"))
    train = train.merge(folds, on="Id", how="left", validate="1:1")
    assert train["fold"].notna().all(), "folds.csv does not match the training data; rerun step1"

    train = engineer(train)
    if test is not None:
        test = engineer(test)
    features = [c for c in train.columns if c not in TARGETS + ["Id", "fold"]]
    cat_cols = [c for c in CAT_COLS if c in features]
    print(f"Features: {len(features)} ({sum(c.startswith('fe_') for c in features)} engineered)")

    n_folds = max(1, min(args.cv_folds, N_FOLDS))
    if args.quick:
        keep = np.random.default_rng(SEED).random(len(train)) < 0.2
        train = train.loc[keep].reset_index(drop=True)
        n_folds = 2
        print(f"QUICK MODE: {len(train):,} rows, {n_folds} folds")

    
    ckdir = _cache(f"checkpoints_xgb_{'quick' if args.quick else 'full'}_{n_folds}folds")
    os.makedirs(ckdir, exist_ok=True)
    print(f"Checkpoints: {ckdir}/" + ("  (--fresh: ignoring saved ones)" if args.fresh else ""))

    rng = np.random.default_rng(SEED)
    cv_rows = train["fold"] >= 0
    oof = pd.DataFrame({"Id": train.loc[cv_rows, "Id"].values}, index=train.index[cv_rows])
    best_iters = {t: [] for t in TARGETS}
    importances = {}
    results = []

    for t in TARGETS:
        t0 = time.time()
        ck = os.path.join(ckdir, f"cv_{t}.pkl")
        if os.path.exists(ck) and not args.fresh:
            with open(ck, "rb") as fh:
                c = pickle.load(fh)
            oof[t], oof[f"{t}_p_pos"] = c["pred"], c["p_pos"]
            best_iters[t], importances[t] = c["best_iters"], c["importance"]
            results.append(c["result"])
            r = c["result"]
            print(f"{t:30s} sMAPE {r['sMAPE']:6.2f}   sign acc {r['sign_acc_%']:5.1f}% "
                  f"(majority {r['majority_sign_%']:5.1f}%)   loaded from checkpoint")
            continue
        p_pos_oof = pd.Series(np.nan, index=oof.index)
        for k in range(n_folds):
            tr = cv_rows & (train["fold"] != k)
            va = train["fold"] == k
            tf = time.time()
            reg, clf = fit_pair(train.loc[tr, features], train.loc[tr], t, cat_cols, rng)
            best_iters[t].append((reg.best_iter, clf.best_iter))
            print(f"    fold {k}: {time.time() - tf:5.0f}s  (rounds: magnitude {reg.best_iter}, "
                  f"sign {clf.best_iter})", flush=True)
            X_va = train.loc[va, features]
            p_pos = clf.predict(X_va)
            oof.loc[va[va].index, t] = combine_xgb(train.loc[va], t, reg.predict(X_va), p_pos)
            p_pos_oof.loc[va[va].index] = p_pos
            imp = reg.importance(features)
            importances[t] = importances.get(t, 0) + imp / n_folds

        idx = oof.index[oof[t].notna()]
        y = train.loc[idx, t].to_numpy(float)
        pred = oof.loc[idx, t].to_numpy(float)
        oof[f"{t}_p_pos"] = p_pos_oof
        sign_acc = 100 * np.mean(np.sign(pred) == np.sign(y))
        majority = 100 * max(np.mean(y > 0), np.mean(y <= 0))
        results.append({"target": t, "sMAPE": smape(y, pred), "sign_acc_%": sign_acc,
                        "majority_sign_%": majority,
                        "mean_iters_reg": np.mean([b[0] for b in best_iters[t]]),
                        "mean_iters_clf": np.mean([b[1] for b in best_iters[t]])})
        print(f"{t:30s} sMAPE {results[-1]['sMAPE']:6.2f}   sign acc {sign_acc:5.1f}% "
              f"(majority {majority:5.1f}%)   {time.time() - t0:5.0f}s", flush=True)
        with open(ck, "wb") as fh:
            pickle.dump({"pred": oof[t], "p_pos": oof[f"{t}_p_pos"], "best_iters": best_iters[t],
                         "importance": importances[t], "result": results[-1]}, fh)

    res = pd.DataFrame(results).set_index("target")
    avg = res["sMAPE"].mean()
    print(f"\nAverage CV sMAPE: {avg:.2f}   (ratio baseline was ~35.6, naive mean ~135.7)")
    print(f"Reduction vs naive mean (135.74): {100 * (1 - avg / 135.74):.1f}%")
    res.loc["AVERAGE"] = [avg] + [np.nan] * (res.shape[1] - 1)
    res.round(3).to_csv(_cache("cv_results_xgb.csv"))
    oof.to_csv(_cache("oof_xgb.csv"), index=False)
    if importances:
        pd.DataFrame(importances).to_csv(_cache("feature_importance_xgb.csv"))
        top = pd.DataFrame(importances).rank(ascending=False).mean(axis=1).sort_values().head(15)
        print("\nTop 15 features by average importance rank:", ", ".join(top.index))

    if args.quick:
        print("\nQuick mode: skipping holdout and final fit.")
        return

    # number of rounds for refits: CV average (x1.1 for the larger final training set)
    iters_cv = {t: (max(1, int(np.mean([b[0] for b in best_iters[t]]))),
                    int(np.mean([b[1] for b in best_iters[t]]))) for t in TARGETS}

    if args.holdout:
        hold = train["fold"] == -1
        pred_h = pd.DataFrame({"Id": train.loc[hold, "Id"].values})
        for t in TARGETS:
            reg, clf = fit_pair(train.loc[cv_rows, features], train.loc[cv_rows], t, cat_cols,
                                rng, n_iter=iters_cv[t])
            X_h = train.loc[hold, features]
            pred_h[t] = combine_xgb(train.loc[hold], t, reg.predict(X_h), clf.predict(X_h))
        scores = {t: smape(train.loc[hold, t], pred_h[t]) for t in TARGETS}
        print("\nHOLDOUT sMAPE:", {t.replace("Q0_", ""): round(s, 2) for t, s in scores.items()})
        print(f"HOLDOUT average: {np.mean(list(scores.values())):.2f}  (CV average was {avg:.2f})")
        pred_h.to_csv(_cache("holdout_xgb.csv"), index=False)

    if test is None:
        print("\nNo test file found; skipping submission.")
        return
    print("\nFitting final models on all training rows...")
    sub = pd.DataFrame({"Id": test["Id"].values})
    for t in TARGETS:
        ck = os.path.join(ckdir, f"final_{t}.pkl")
        if os.path.exists(ck) and not args.fresh:
            with open(ck, "rb") as fh:
                sub[t] = pickle.load(fh)
            print(f"  {t} loaded from checkpoint")
            continue
        tf = time.time()
        it = tuple(int(i * FINAL_ITER_MULT) for i in iters_cv[t])
        reg, clf = fit_pair(train[features], train, t, cat_cols, rng, n_iter=it)
        X_te = test[features]
        sub[t] = combine_xgb(test, t, reg.predict(X_te), clf.predict(X_te))
        with open(ck, "wb") as fh:
            pickle.dump(sub[t].to_numpy(), fh)
        print(f"  {t} done ({time.time() - tf:.0f}s)", flush=True)
    assert list(sub.columns) == ["Id"] + TARGETS and len(sub) == len(test)
    assert np.isfinite(sub[TARGETS].to_numpy()).all() and (sub[TARGETS] != 0).all().all()
    sub.to_csv("submission_xgb.csv", index=False)
    print(f"Wrote submission_xgb.csv ({len(sub):,} rows)")



"""
Step 3: combine predictions using the Q0 identities

No retraining. Uses the out-of-fold predictions from step 2 to choose, for each target,
a weighted average of its direct prediction and estimates derived from other targets:
    EBITDA  ~ OPERATING_INCOME + Q1 D&A          OPERATING_INCOME = GROSS_PROFIT - OPERATING_EXPENSES
    EQUITY  = ASSETS - LIABILITIES               GROSS_PROFIT ~ REVENUES - COST_OF_REVENUES
Weights are picked on a 0.05 grid. Nested CV (choose weights on 4 folds, score the 5th)
gives an honest estimate; a target keeps its direct prediction unless blending helps.
"""

GRID_STEP = 0.05
DA = "Q1_DEPRECIATION_AND_AMORTIZATION"


def candidates(P, raw):
    """P: predictions (columns = target names). raw: rows with Q1 D&A. Same row order.
    Returns {target: [(label, values), ...]}; the first entry is always the direct prediction."""
    g = lambda t: P[f"Q0_{t}"].to_numpy(float)
    da = raw[DA].to_numpy(float)
    return {
        "Q0_TOTAL_ASSETS": [("direct", g("TOTAL_ASSETS")),
                            ("L+E", g("TOTAL_LIABILITIES") + g("TOTAL_STOCKHOLDERS_EQUITY"))],
        "Q0_TOTAL_LIABILITIES": [("direct", g("TOTAL_LIABILITIES")),
                                 ("A-E", g("TOTAL_ASSETS") - g("TOTAL_STOCKHOLDERS_EQUITY"))],
        "Q0_TOTAL_STOCKHOLDERS_EQUITY": [("direct", g("TOTAL_STOCKHOLDERS_EQUITY")),
                                         ("A-L", g("TOTAL_ASSETS") - g("TOTAL_LIABILITIES"))],
        "Q0_GROSS_PROFIT": [("direct", g("GROSS_PROFIT")),
                            ("Rev-COGS", g("REVENUES") - g("COST_OF_REVENUES")),
                            ("OI+OpEx", g("OPERATING_INCOME") + g("OPERATING_EXPENSES"))],
        "Q0_COST_OF_REVENUES": [("direct", g("COST_OF_REVENUES")),
                                ("Rev-GP", g("REVENUES") - g("GROSS_PROFIT"))],
        "Q0_REVENUES": [("direct", g("REVENUES")),
                        ("GP+COGS", g("GROSS_PROFIT") + g("COST_OF_REVENUES"))],
        "Q0_OPERATING_INCOME": [("direct", g("OPERATING_INCOME")),
                                ("EBITDA-DA", g("EBITDA") - da),
                                ("GP-OpEx", g("GROSS_PROFIT") - g("OPERATING_EXPENSES"))],
        "Q0_OPERATING_EXPENSES": [("direct", g("OPERATING_EXPENSES")),
                                  ("GP-OI", g("GROSS_PROFIT") - g("OPERATING_INCOME"))],
        "Q0_EBITDA": [("direct", g("EBITDA")),
                      ("OI+DA", g("OPERATING_INCOME") + da),
                      ("GP-OpEx+DA", g("GROSS_PROFIT") - g("OPERATING_EXPENSES") + da)],
    }


def apply_weights(cands, w):
    C = np.column_stack([v for _, v in cands])
    p = C @ w
    direct = C[:, 0]
    return np.where(np.abs(p) < 1, direct, p)  # never output ~0 (scores 200%)


def term_errors(y, p):
    denom = (np.abs(y) + np.abs(p)) / 2
    return 100 * np.abs(y - p) / np.where(denom == 0, 1, denom)


def run_step3():
    train, test, _ = load_data()
    oof = pd.read_csv(_cache("oof_xgb.csv"))
    folds = pd.read_csv(_cache("folds.csv"))
    cols = ["Id", DA] + TARGETS
    d = oof[["Id"] + TARGETS].merge(train[cols], on="Id", suffixes=("_hat", ""), validate="1:1")
    d = d.merge(folds, on="Id", validate="1:1")
    assert len(d) == len(oof) and (d["fold"] >= 0).all(), "oof/folds/train do not line up"
    P = d[[f"{t}_hat" for t in TARGETS]].set_axis(TARGETS, axis=1)
    fold = d["fold"].to_numpy()
    cands_oof = candidates(P, d)

    rows, weights = [], {}
    for t in TARGETS:
        cands = cands_oof[t]
        n = len(cands)
        W = np.array([w for w in itertools.product(np.arange(0, 1 + 1e-9, GRID_STEP), repeat=n)
                      if abs(sum(w) - 1) < 1e-9])
        y = d[t].to_numpy(float)
        errs = np.stack([term_errors(y, apply_weights(cands, w)) for w in W], axis=1)
        direct_i = int(np.where(np.isclose(W, np.eye(n)[0]).all(axis=1))[0][0])

        nested = np.zeros(len(y))
        for k in np.unique(fold):
            best = errs[fold != k].mean(axis=0).argmin()
            nested[fold == k] = errs[fold == k, best]
        direct_score, nested_score = errs[:, direct_i].mean(), nested.mean()
        w_full = W[errs.mean(axis=0).argmin()] if nested_score < direct_score else W[direct_i]
        weights[t] = w_full
        rows.append({"target": t, "direct_sMAPE": direct_score, "blend_sMAPE_nested": nested_score,
                     "weights": ", ".join(f"{lab} {w:.2f}" for (lab, _), w in zip(cands, w_full))})

    rep = pd.DataFrame(rows).set_index("target")
    pd.set_option("display.width", 200)
    print("\nOut-of-fold sMAPE (blend scored with nested CV):")
    print(rep.round(2).to_string())
    print(f"\nAverage: direct {rep.direct_sMAPE.mean():.2f} -> blended {rep.blend_sMAPE_nested.mean():.2f}")
    rep.to_csv(_cache("blend_weights.csv"))

    if test is None:
        print("No test file found; skipping submission.")
        return
    sub = pd.read_csv("submission_xgb.csv")
    assert list(sub.columns) == ["Id"] + TARGETS, "submission_xgb.csv has unexpected columns"
    raw = sub[["Id"]].merge(test[["Id", DA]], on="Id", how="left", validate="1:1")
    assert raw[DA].notna().all(), "submission Ids do not match the test file"
    cands_test = candidates(sub[TARGETS], raw)       # built from the ORIGINAL predictions
    out = sub[["Id"]].copy()
    for t in TARGETS:
        out[t] = apply_weights(cands_test[t], weights[t])
    assert np.isfinite(out[TARGETS].to_numpy()).all() and (out[TARGETS] != 0).all().all()
    out.to_csv("submission_xgb_blend.csv", index=False)
    changed = {t.replace("Q0_", ""): f"{100 * np.mean(np.sign(out[t]) != np.sign(sub[t])):.2f}%"
               for t in TARGETS}
    print("\nShare of test rows whose sign changed vs submission_xgb.csv:", changed)
    print(f"Wrote submission_xgb_blend.csv ({len(out):,} rows)")



"""
Step 4: GRU-based neural network

Model
  * Sequence branch: quarters Q10 -> Q1 in time order. Each step has the 16 financial
    fields (signed log, standardized), that quarter's fiscal-year-end flag, and a flag
    marking Q3/Q7 (whose balance-sheet values look altered) so the GRU can learn to
    discount them. 2-layer GRU; last output + mean of outputs are kept.
  * Static branch (dense): company-profile columns, Q0_fiscal_year_end, placeholder flags,
    and embeddings for industry / sector / currency / rating. Use --no-static for a
    sequence-only model.
  * One network predicts all 9 targets, with two outputs per target (same idea as XGBoost):
      magnitude: log1p|Q0| - log1p|Q1|   (smooth L1 loss, L1-like, aligned with sMAPE)
      sign:      logit of P(Q0 > 0)       (binary cross-entropy)
    Prediction = sign * max(magnitude, 1).
  * Early stopping on the average sMAPE of a 10% slice of each training fold.

Outputs (same format as the XGBoost files, so they can be blended):
  cv_results_gru.csv, oof_gru.csv, submission_gru.csv, holdout_gru.csv
"""

# ------------------------------------------------------------------ settings
HIDDEN = 128
GRU_LAYERS = 2
DROPOUT = 0.2
BATCH = 512
LR = 2e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
PATIENCE = 6              # epochs without improvement in early-stopping sMAPE
ES_FRAC = 0.10
SIGN_LOSS_WEIGHT = 0.5
HUBER_BETA = 0.2          # smooth L1: quadratic below 0.2 on the log scale, linear above
MAG_CLIP = 8.0            # clip extreme log-ratio training targets
FINAL_EPOCH_MULT = 1.10

FIELDS = ["TOTAL_ASSETS", "TOTAL_CURRENT_ASSETS", "TOTAL_NONCURRENT_ASSETS",
          "TOTAL_LIABILITIES", "TOTAL_CURRENT_LIABILITIES", "TOTAL_NONCURRENT_LIABILITIES",
          "TOTAL_LIABILITIES_AND_EQUITY", "TOTAL_STOCKHOLDERS_EQUITY", "NET_INCOME",
          "GROSS_PROFIT", "COST_OF_REVENUES", "REVENUES", "OPERATING_INCOME",
          "OPERATING_EXPENSES", "EBITDA", "DEPRECIATION_AND_AMORTIZATION"]
QUARTERS = list(range(10, 0, -1))   # Q10 (oldest) ... Q1 (most recent)
ALTERED = {3, 7}
LAG_COL = re.compile(r"^Q([1-9]|10)_")


def slog(x):
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------ data -> arrays
def static_frame(df, features):
    """Non-lag numeric predictors + a few placeholder flags (raw values are not modified)."""
    cols = [c for c in features if not LAG_COL.match(c) and c not in CAT_COLS]
    s = df[cols].astype(float).copy()
    for c in ["auditRisk", "boardRisk", "compensationRisk", "shareHolderRightsRisk", "overallRisk"]:
        if c in df:
            s[f"fill_{c}"] = (df[c] == 6.0).astype(float)
    s["fill_targetprices"] = ((df["targetHighPrice"] == 30) & (df["targetLowPrice"] == 17)).astype(float)
    s["fill_analysts"] = (df["numberOfAnalystOpinions"] == 6).astype(float)
    s["fill_forwardEps"] = (df["forwardEps"] == 0.32).astype(float)
    return s


def build_arrays(df, features):
    steps = []
    for k in QUARTERS:
        cols = [slog(df[f"Q{k}_{f}"]) for f in FIELDS]
        cols.append(df[f"Q{k}_fiscal_year_end"].to_numpy(float))
        cols.append(np.full(len(df), 1.0 if k in ALTERED else 0.0))
        steps.append(np.column_stack(cols))
    seq = np.stack(steps, axis=1).astype(np.float32)                       # n x 10 x 18
    stat = slog(static_frame(df, features).to_numpy(float)).astype(np.float32)
    cats = np.column_stack([df[c].cat.codes.to_numpy() + 1 for c in CAT_COLS]).astype(np.int64)
    base = np.column_stack([np.log1p(np.abs(df[t.replace("Q0_", "Q1_")].to_numpy(float)))
                            for t in TARGETS]).astype(np.float32)          # log1p|Q1| per target
    return {"seq": seq, "stat": stat, "cats": cats, "base": base}


def targets_arrays(df, base):
    y = df[TARGETS].to_numpy(float)
    mag = np.clip(np.log1p(np.abs(y)) - base, -MAG_CLIP, MAG_CLIP).astype(np.float32)
    pos = (y > 0).astype(np.float32)
    return y, mag, pos


class Scaler:
    """Standardizes sequence features per field and static features per column (fit on train rows)."""

    def fit(self, A):
        self.sm = np.nanmean(A["seq"], axis=(0, 1))
        self.ss = np.nanstd(A["seq"], axis=(0, 1)) + 1e-6
        self.tm = np.nanmean(A["stat"], axis=0)
        self.ts = np.nanstd(A["stat"], axis=0) + 1e-6
        return self

    def transform(self, A):
        out = dict(A)
        out["seq"] = np.clip(np.nan_to_num((A["seq"] - self.sm) / self.ss), -6, 6).astype(np.float32)
        out["stat"] = np.clip(np.nan_to_num((A["stat"] - self.tm) / self.ts), -6, 6).astype(np.float32)
        return out


def subset(A, idx):
    return {k: v[idx] for k, v in A.items()}


# ------------------------------------------------------------------ model
class GRUNet(nn.Module):
    def __init__(self, n_seq, n_stat, cat_sizes, use_static=True):
        super().__init__()
        self.use_static = use_static
        self.inp = nn.Sequential(nn.Linear(n_seq, HIDDEN), nn.GELU())
        self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=GRU_LAYERS, batch_first=True,
                          dropout=DROPOUT if GRU_LAYERS > 1 else 0.0)
        trunk_in = 2 * HIDDEN
        if use_static:
            self.embs = nn.ModuleList([nn.Embedding(n + 1, min(10, (n + 2) // 2)) for n in cat_sizes])
            emb_dim = sum(e.embedding_dim for e in self.embs)
            self.stat = nn.Sequential(nn.Linear(n_stat + emb_dim, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT))
            trunk_in += HIDDEN
        self.trunk = nn.Sequential(nn.Linear(trunk_in, 256), nn.GELU(), nn.Dropout(DROPOUT),
                                   nn.Linear(256, 128), nn.GELU())
        self.mag = nn.Linear(128, len(TARGETS))
        self.sign = nn.Linear(128, len(TARGETS))

    def forward(self, seq, stat, cats):
        out, _ = self.gru(self.inp(seq))
        parts = [out[:, -1], out.mean(dim=1)]
        if self.use_static:
            emb = [e(cats[:, i]) for i, e in enumerate(self.embs)]
            parts.append(self.stat(torch.cat([stat] + emb, dim=1)))
        h = self.trunk(torch.cat(parts, dim=1))
        return self.mag(h), self.sign(h)


def to_t(A, idx, device):
    return (torch.from_numpy(A["seq"][idx]).to(device), torch.from_numpy(A["stat"][idx]).to(device),
            torch.from_numpy(A["cats"][idx]).to(device))


@torch.no_grad()
def predict(model, A, device):
    model.eval()
    mags, probs = [], []
    for i in range(0, len(A["seq"]), 4096):
        idx = np.arange(i, min(i + 4096, len(A["seq"])))
        m, s = model(*to_t(A, idx, device))
        mags.append(m.float().cpu().numpy())
        probs.append(torch.sigmoid(s).float().cpu().numpy())
    return np.vstack(mags), np.vstack(probs)


def combine_gru(mag, p_pos, base):
    magnitude = np.expm1(np.clip(mag + base, None, 60))
    return np.where(p_pos >= 0.5, 1.0, -1.0) * np.maximum(magnitude, 1.0)


def avg_smape(y, pred):
    return float(np.mean([smape(y[:, j], pred[:, j]) for j in range(len(TARGETS))]))


def train_net(A_tr, mag_tr, pos_tr, cat_sizes, device, seed, use_static,
              A_es=None, y_es=None, n_epochs=None, max_epochs=MAX_EPOCHS, log_prefix=""):
    """With A_es: early stopping on ES sMAPE (returns best epoch). Without: trains n_epochs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GRUNet(A_tr["seq"].shape[2], A_tr["stat"].shape[1], cat_sizes, use_static).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # same cosine schedule in CV and in the final fit, so "epoch k" means the same thing
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=LR / 50)
    huber = nn.SmoothL1Loss(beta=HUBER_BETA)
    bce = nn.BCEWithLogitsLoss()
    mag_t, pos_t = torch.from_numpy(mag_tr).to(device), torch.from_numpy(pos_tr).to(device)

    epochs = n_epochs if A_es is None else max_epochs
    best, best_epoch, best_state, bad = np.inf, 0, None, 0
    n = len(mag_tr)
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        order = rng.permutation(n)
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            seq, stat, cats = to_t(A_tr, idx, device)
            m, s = model(seq, stat, cats)
            loss = huber(m, mag_t[idx]) + SIGN_LOSS_WEIGHT * bce(s, pos_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if A_es is None:
            print(f"{log_prefix}epoch {ep:2d}/{epochs}  loss {tot / n:.4f}  {time.time() - t0:4.0f}s", flush=True)
            continue
        mag_es, p_es = predict(model, A_es, device)
        score = avg_smape(y_es, combine_gru(mag_es, p_es, A_es["base"]))
        flag = ""
        if score < best - 1e-4:
            best, best_epoch, bad, flag = score, ep, 0, " *"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"{log_prefix}epoch {ep:2d}  loss {tot / n:.4f}  ES sMAPE {score:6.2f}{flag}  "
              f"{time.time() - t0:4.0f}s", flush=True)
        if bad >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_epoch if A_es is not None else epochs)


def fit_and_predict(df_tr, df_pred_list, features, cat_sizes, device, seed, use_static,
                    early_stop=True, n_epochs=None, max_epochs=MAX_EPOCHS, log_prefix=""):
    """Builds arrays, fits the scaler on df_tr only, trains, predicts each frame in df_pred_list."""
    A_tr_raw = build_arrays(df_tr, features)
    rng = np.random.default_rng(seed)
    if early_stop:
        es = rng.random(len(df_tr)) < ES_FRAC
        scaler = Scaler().fit(subset(A_tr_raw, ~es))
        A_all = scaler.transform(A_tr_raw)
        y_all, mag_all, pos_all = targets_arrays(df_tr, A_tr_raw["base"])
        model, best_ep = train_net(subset(A_all, ~es), mag_all[~es], pos_all[~es], cat_sizes, device,
                                   seed, use_static, A_es=subset(A_all, es), y_es=y_all[es],
                                   max_epochs=max_epochs, log_prefix=log_prefix)
    else:
        scaler = Scaler().fit(A_tr_raw)
        A_all = scaler.transform(A_tr_raw)
        _, mag_all, pos_all = targets_arrays(df_tr, A_tr_raw["base"])
        model, best_ep = train_net(A_all, mag_all, pos_all, cat_sizes, device, seed, use_static,
                                   n_epochs=n_epochs, max_epochs=max_epochs, log_prefix=log_prefix)
    outs = []
    for dfp in df_pred_list:
        A = scaler.transform(build_arrays(dfp, features))
        mag, p = predict(model, A, device)
        outs.append((combine_gru(mag, p, A["base"]), p))
    return outs, best_ep


# ------------------------------------------------------------------ main
def run_step4(args):
    use_static = not args.no_static

    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    print(f"PyTorch {torch.__version__}, device: {device}, static branch: {use_static}")

    train, test, features = load_data()
    folds = pd.read_csv(_cache("folds.csv"))
    train = train.merge(folds, on="Id", how="left", validate="1:1")
    assert train["fold"].notna().all(), "folds.csv does not match the training data; rerun step1"
    cat_sizes = [len(train[c].cat.categories) for c in CAT_COLS]

    n_folds, max_epochs = N_FOLDS, MAX_EPOCHS
    if args.quick:
        keep = np.random.default_rng(SEED).random(len(train)) < 0.2
        train = train.loc[keep].reset_index(drop=True)
        n_folds, max_epochs = 2, 4
        print(f"QUICK MODE: {len(train):,} rows, {n_folds} folds, {max_epochs} epochs")

    tag = ("quick" if args.quick else "full") + ("" if use_static else "_seqonly")
    ckdir = _cache(f"checkpoints_gru_{tag}")
    os.makedirs(ckdir, exist_ok=True)
    print(f"Checkpoints: {ckdir}/" + ("  (--fresh: ignoring saved ones)" if args.fresh else ""))

    cv_rows = (train["fold"] >= 0).to_numpy()
    oof_pred = np.full((len(train), len(TARGETS)), np.nan)
    oof_p = np.full((len(train), len(TARGETS)), np.nan)
    best_epochs = []
    for k in range(n_folds):
        ck = os.path.join(ckdir, f"fold_{k}.pkl")
        va = (train["fold"] == k).to_numpy()
        if os.path.exists(ck) and not args.fresh:
            with open(ck, "rb") as fh:
                c = pickle.load(fh)
            oof_pred[va], oof_p[va] = c["pred"], c["p"]
            best_epochs.append(c["best_epoch"])
            print(f"fold {k}: loaded from checkpoint (best epoch {c['best_epoch']})")
            continue
        tr = cv_rows & (train["fold"] != k).to_numpy()
        t0 = time.time()
        print(f"fold {k}: training on {tr.sum():,} rows")
        [(pred, p)], best_ep = fit_and_predict(train.loc[tr], [train.loc[va]], features, cat_sizes,
                                               device, SEED + k, use_static, max_epochs=max_epochs,
                                               log_prefix="    ")
        oof_pred[va], oof_p[va] = pred, p
        best_epochs.append(best_ep)
        fold_score = avg_smape(train.loc[va, TARGETS].to_numpy(float), pred)
        print(f"fold {k}: sMAPE {fold_score:.2f}, best epoch {best_ep}, {time.time() - t0:.0f}s", flush=True)
        with open(ck, "wb") as fh:
            pickle.dump({"pred": pred, "p": p, "best_epoch": best_ep}, fh)

    done = ~np.isnan(oof_pred[:, 0])
    y = train.loc[done, TARGETS].to_numpy(float)
    rows = []
    for j, t in enumerate(TARGETS):
        pr = oof_pred[done, j]
        rows.append({"target": t, "sMAPE": smape(y[:, j], pr),
                     "sign_acc_%": 100 * np.mean(np.sign(pr) == np.sign(y[:, j])),
                     "majority_sign_%": 100 * max(np.mean(y[:, j] > 0), np.mean(y[:, j] <= 0))})
        print(f"{t:30s} sMAPE {rows[-1]['sMAPE']:6.2f}   sign acc {rows[-1]['sign_acc_%']:5.1f}% "
              f"(majority {rows[-1]['majority_sign_%']:5.1f}%)")
    res = pd.DataFrame(rows).set_index("target")
    avg = res["sMAPE"].mean()
    print(f"\nAverage CV sMAPE (GRU): {avg:.2f}   (XGBoost was 13.01; naive mean ~135.7)")
    res.loc["AVERAGE"] = [avg, np.nan, np.nan]
    res.round(3).to_csv(_cache("cv_results_gru.csv"))
    oof = pd.DataFrame({"Id": train.loc[done, "Id"].values})
    for j, t in enumerate(TARGETS):
        oof[t] = oof_pred[done, j]
        oof[f"{t}_p_pos"] = oof_p[done, j]
    oof.to_csv(_cache("oof_gru.csv"), index=False)
    print("Wrote cv_results_gru.csv, oof_gru.csv")

    if args.quick:
        print("Quick mode: skipping holdout and final fit.")
        return
    n_final = max(1, int(round(np.mean(best_epochs) * FINAL_EPOCH_MULT)))

    if args.holdout:
        hold = (train["fold"] == -1).to_numpy()
        n_cv = max(1, int(round(np.mean(best_epochs))))
        print(f"\nHoldout: training on CV rows for {n_cv} epochs")
        [(pred_h, _)], _ = fit_and_predict(train.loc[cv_rows], [train.loc[hold]], features, cat_sizes,
                                           device, SEED + 100, use_static, early_stop=False,
                                           n_epochs=n_cv, max_epochs=max_epochs, log_prefix="    ")
        yh = train.loc[hold, TARGETS].to_numpy(float)
        scores = {t.replace("Q0_", ""): round(smape(yh[:, j], pred_h[:, j]), 2) for j, t in enumerate(TARGETS)}
        print("HOLDOUT sMAPE:", scores)
        print(f"HOLDOUT average: {avg_smape(yh, pred_h):.2f}  (CV average was {avg:.2f})")
        pd.DataFrame(pred_h, columns=TARGETS).assign(Id=train.loc[hold, "Id"].values)[["Id"] + TARGETS] \
            .to_csv(_cache("holdout_gru.csv"), index=False)

    if test is None:
        print("No test file found; skipping submission.")
        return
    ck = os.path.join(ckdir, "final.pkl")
    if os.path.exists(ck) and not args.fresh:
        with open(ck, "rb") as fh:
            pred_te, p_te = pickle.load(fh)
        print("\nFinal model: loaded predictions from checkpoint")
    else:
        print(f"\nFinal model: training on all {len(train):,} rows for {n_final} epochs")
        [(pred_te, p_te)], _ = fit_and_predict(train, [test], features, cat_sizes, device, SEED + 999,
                                               use_static, early_stop=False, n_epochs=n_final,
                                               max_epochs=max_epochs, log_prefix="    ")
        with open(ck, "wb") as fh:
            pickle.dump((pred_te, p_te), fh)
    sub = pd.DataFrame(pred_te, columns=TARGETS)
    sub.insert(0, "Id", test["Id"].values)
    assert np.isfinite(sub[TARGETS].to_numpy()).all() and (sub[TARGETS] != 0).all().all()
    sub.to_csv("submission_gru.csv", index=False)
    # test sign probabilities, in case they are useful for blending later
    pd.DataFrame(p_te, columns=[f"{t}_p_pos" for t in TARGETS]).assign(Id=test["Id"].values) \
        .to_csv(_cache("test_sign_probs_gru.csv"), index=False)
    print(f"Wrote submission_gru.csv ({len(sub):,} rows)")



"""
Step 5: XGBoost + GRU hybrid

No retraining. For each target:
  1. rebuilds the identity-blended XGBoost prediction (same method and weights as step 3)
  2. chooses a weight w for the GRU on a 0.05 grid:  final = (1 - w) * XGB_blend + w * GRU
Weights are chosen on out-of-fold predictions. A target keeps w = 0 unless adding the GRU helps.
"""

def choose_weights(cands, y, fold):
    """Grid search over weights summing to 1. Returns (full-data weights, direct score, nested score).
    The first candidate is the 'keep as is' option; it is kept unless the nested score improves."""
    n = len(cands)
    W = np.array([w for w in itertools.product(np.arange(0, 1 + 1e-9, GRID_STEP), repeat=n)
                  if abs(sum(w) - 1) < 1e-9])
    errs = np.stack([term_errors(y, apply_weights(cands, w)) for w in W], axis=1)
    first = int(np.where(np.isclose(W, np.eye(n)[0]).all(axis=1))[0][0])
    nested = np.zeros(len(y))
    for k in np.unique(fold):
        best = errs[fold != k].mean(axis=0).argmin()
        nested[fold == k] = errs[fold == k, best]
    direct, nest = errs[:, first].mean(), nested.mean()
    w = W[errs.mean(axis=0).argmin()] if nest < direct else W[first]
    return w, direct, nest


def run_step5(args):
    train, test, _ = load_data()
    folds = pd.read_csv(_cache("folds.csv"))
    train = train.merge(folds, on="Id", validate="1:1")  # adds train["fold"], used by the holdout check below
    ox = pd.read_csv(_cache("oof_xgb.csv"))[["Id"] + TARGETS]
    og = pd.read_csv(_cache("oof_gru.csv"))[["Id"] + TARGETS]
    d = (ox.merge(og, on="Id", suffixes=("", "_gru"), validate="1:1")
           .merge(train[["Id", DA] + TARGETS], on="Id", suffixes=("", "_true"), validate="1:1")
           .merge(folds, on="Id", validate="1:1"))
    assert len(d) == len(ox) == len(og) and (d["fold"] >= 0).all(), "OOF files / folds do not line up"
    fold = d["fold"].to_numpy()
    P_xgb = d[TARGETS]
    cands_xgb = candidates(P_xgb, d)

    rows, w_id, w_gru = [], {}, {}
    for t in TARGETS:
        y = d[f"{t}_true"].to_numpy(float)
        # step 1: identity blend of the XGBoost predictions (reproduces step 3)
        w_id[t], xgb_direct, _ = choose_weights(cands_xgb[t], y, fold)
        xgb_blend = apply_weights(cands_xgb[t], w_id[t])
        # step 2: add the GRU
        gru = d[f"{t}_gru"].to_numpy(float)
        w, blend_score, hybrid_nested = choose_weights([("xgb_blend", xgb_blend), ("gru", gru)], y, fold)
        w_gru[t] = w[1]
        rows.append({"target": t, "xgb_direct": xgb_direct, "xgb_blend": blend_score,
                     "gru_alone": term_errors(y, gru).mean(), "hybrid_nested": hybrid_nested,
                     "gru_weight": w[1]})

    rep = pd.DataFrame(rows).set_index("target")
    pd.set_option("display.width", 200)
    print("\nOut-of-fold sMAPE (hybrid scored with nested CV):")
    print(rep.round(3).to_string())
    avg = rep.mean()
    print(f"\nAverage: XGB {avg.xgb_direct:.3f} -> XGB blend {avg.xgb_blend:.3f} -> "
          f"hybrid {avg.hybrid_nested:.3f}   (GRU alone {avg.gru_alone:.3f})")
    rep.to_csv(_cache("hybrid_weights.csv"))

    # True-holdout check: w_id/w_gru above were chosen entirely from CV folds (fold >= 0), so the
    # holdout rows (fold == -1) are still untouched here - this is an honest, non-nested estimate
    # of whether the hybrid actually beats plain XGBoost, XGBoost's identity blend and the GRU.
    if args.holdout:
        hx_path, hg_path = _cache("holdout_xgb.csv"), _cache("holdout_gru.csv")
        if not (os.path.exists(hx_path) and os.path.exists(hg_path)):
            print("\n--holdout was set, but holdout_xgb.csv/holdout_gru.csv are missing from the "
                  "cache (they're only written when steps 2 and 4 are ALSO run with --holdout) - "
                  "skipping the true-holdout comparison.")
        else:
            hx = pd.read_csv(hx_path)                                                    # Id + TARGETS
            hg = hx[["Id"]].merge(pd.read_csv(hg_path), on="Id", how="left", validate="1:1")
            hold = hx[["Id"]].merge(train, on="Id", how="left", validate="1:1")           # full columns, aligned
            assert hg[TARGETS].notna().all().all() and hold[TARGETS + [DA]].notna().all().all(), \
                "holdout_xgb.csv / holdout_gru.csv / train do not share the same Ids"

            ratio_params = {t: fit_ratio(train.loc[train["fold"] != -1], t) for t in TARGETS}
            cands_hold = candidates(hx[TARGETS], hold)
            hrows = []
            for t in TARGETS:
                y = hold[t].to_numpy(float)
                xgb_blend = apply_weights(cands_hold[t], w_id[t])
                gru_only = hg[t].to_numpy(float)
                w = np.array([1 - w_gru[t], w_gru[t]])
                hybrid = apply_weights([("xgb_blend", xgb_blend), ("gru", gru_only)], w)
                hrows.append({"target": t,
                              "ratio_baseline": smape(y, predict_ratio(hold, t, ratio_params[t])),
                              "xgb_direct": smape(y, hx[t]), "xgb_blend": smape(y, xgb_blend),
                              "gru_alone": smape(y, gru_only), "hybrid": smape(y, hybrid)})
            hrep = pd.DataFrame(hrows).set_index("target")
            havg = hrep.mean()
            pd.set_option("display.width", 200)
            print(f"\nTRUE HOLDOUT sMAPE ({len(hx):,} rows never used for OOF, CV, or weight selection):")
            print(hrep.round(3).to_string())
            print(f"\nHoldout average: ratio {havg.ratio_baseline:.3f} -> XGB {havg.xgb_direct:.3f} -> "
                  f"XGB blend {havg.xgb_blend:.3f} -> hybrid {havg.hybrid:.3f}   "
                  f"(GRU alone {havg.gru_alone:.3f})")
            hrep.to_csv(_cache("hybrid_holdout.csv"))

    if test is None:
        print("No test file found; skipping submission.")
        return
    sx = pd.read_csv("submission_xgb.csv")
    sg = pd.read_csv("submission_gru.csv")
    for name, s in [("submission_xgb.csv", sx), ("submission_gru.csv", sg)]:
        assert list(s.columns) == ["Id"] + TARGETS, f"{name} has unexpected columns"
    sg = sx[["Id"]].merge(sg, on="Id", how="left", validate="1:1")
    raw = sx[["Id"]].merge(test[["Id", DA]], on="Id", how="left", validate="1:1")
    assert sg[TARGETS].notna().all().all() and raw[DA].notna().all(), "Ids do not match across files"

    cands_test = candidates(sx[TARGETS], raw)
    out = sx[["Id"]].copy()
    for t in TARGETS:
        xgb_blend = apply_weights(cands_test[t], w_id[t])
        w = np.array([1 - w_gru[t], w_gru[t]])
        out[t] = apply_weights([("xgb_blend", xgb_blend), ("gru", sg[t].to_numpy(float))], w)

    # consistency check against step 3's file, if present
    if os.path.exists("submission_xgb_blend.csv"):
        sb = sx[["Id"]].merge(pd.read_csv("submission_xgb_blend.csv"), on="Id", how="left")
        same = all(np.allclose(apply_weights(cands_test[t], w_id[t]), sb[t], rtol=1e-6) for t in TARGETS)
        print("XGB blend matches submission_xgb_blend.csv:", same)

    assert np.isfinite(out[TARGETS].to_numpy()).all() and (out[TARGETS] != 0).all().all()
    assert len(out) == len(test) and list(out.columns) == ["Id"] + TARGETS
    out.to_csv("submission_hybrid.csv", index=False)
    print(f"Wrote submission_hybrid.csv ({len(out):,} rows)")



if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="46-937 individual prediction pipeline (steps 1-5).",
    )
    ap.add_argument("--quick", action="store_true",
                     help="smoke test: fewer rows/folds/epochs for XGBoost and the GRU; "
                          "steps 3 and 5 are skipped since they need the full submission files")
    ap.add_argument("--holdout", action="store_true",
                     help="also score the 10%% final holdout (do this ONCE, at the very end)")
    ap.add_argument("--gpu", action="store_true", help="train XGBoost on an NVIDIA GPU (device='cuda')")
    ap.add_argument("--cv-folds", type=int, default=N_FOLDS,
                     help="how many of the 5 folds XGBoost evaluates (fewer = faster, noisier CV)")
    ap.add_argument("--fresh", action="store_true",
                     help="ignore any saved checkpoints under _pipeline_cache/ and retrain everything")
    ap.add_argument("--no-static", action="store_true",
                     help="GRU: sequence branch only, no dense static branch")
    args = ap.parse_args()

    
    run_step1()
    run_step2(args)
    run_step3()
    run_step4(args)
    run_step5(args)

    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    made = [f for f in ["submission_ratio_baseline.csv", "submission_xgb.csv",
                         "submission_xgb_blend.csv", "submission_gru.csv", "submission_hybrid.csv"]
            if os.path.exists(f)]
    print(f"\nDone. Removed {CACHE_DIR}/. Submission files present: {', '.join(made)}")