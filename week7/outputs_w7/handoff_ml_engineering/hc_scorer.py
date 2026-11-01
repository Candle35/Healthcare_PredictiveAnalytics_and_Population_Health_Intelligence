"""HealthConnect admission-proxy scorer: validated input -> scored output.

Wraps the refined Data Science model with the input contract agreed for ML Engineering.
Hard problems (missing columns, non-numeric or out-of-range values) are reported, never silently scored.
Soft problems (missing/unseen category, imputed number) are scored but flagged in `status`.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib, sklearn


class SchemaError(ValueError):
    """Raised when the input frame cannot satisfy the contract (e.g. a required column is absent)."""


class AdmissionProxyScorer:
    MODEL_STATUS = "EXPERIMENTAL - NO PREDICTIVE SIGNAL (release gate FAILED); not for clinical or operational decisions"

    def __init__(self, folder="."):
        folder = Path(folder)
        self.schema = json.loads((folder / "schema.json").read_text())
        self.model = joblib.load(folder / "model.joblib")
        if self.schema["sklearn_version"] != sklearn.__version__:
            warnings.warn(f"scikit-learn {sklearn.__version__} != training version {self.schema['sklearn_version']}; results may differ")
        self.required = list(self.schema["numeric"]) + list(self.schema["categorical"])
        self._canon = {c: {str(v).lower(): v for v in spec["categories"]} for c, spec in self.schema["categorical"].items()}

    def validate(self, df):
        """Return (clean_frame, hard_reasons, soft_reasons, extra_columns). Lists are per row (positional)."""
        missing = [c for c in self.required if c not in df.columns]
        if missing:
            raise SchemaError(f"Missing required column(s): {missing}. Expected: {self.required}")
        extra = [c for c in df.columns if c not in self.required]
        d = df[self.required].reset_index(drop=True).copy()
        hard = [[] for _ in range(len(d))]; soft = [[] for _ in range(len(d))]
        for c, spec in self.schema["numeric"].items():
            raw = d[c]; v = pd.to_numeric(raw, errors="coerce")
            for i in np.where(raw.notna() & v.isna())[0]: hard[i].append(f"{c} not numeric")
            for i in np.where(v.notna() & ((v < spec["min"]) | (v > spec["max"])))[0]: hard[i].append(f"{c} out of range [{spec['min']},{spec['max']}]")
            for i in np.where(raw.isna())[0]: soft[i].append(f"{c} missing (imputed)")
            d[c] = v
        for c in self.schema["categorical"]:
            out = []
            for i, x in enumerate(d[c].tolist()):
                if x is None or (isinstance(x, float) and np.isnan(x)) or (not isinstance(x, str) and pd.isna(x)):
                    soft[i].append(f"{c} missing"); out.append("UNKNOWN")
                else:
                    k = str(x).strip().lower()
                    if k in self._canon[c]: out.append(self._canon[c][k])
                    else: soft[i].append(f"{c} unseen category '{x}'"); out.append("UNKNOWN")
            d[c] = out
        return d, hard, soft, extra

    def score(self, df):
        cols = ["risk_score", "flag_top20", "status", "model_status"]
        if len(df) == 0:
            return pd.DataFrame(columns=cols, index=df.index)
        d, hard, soft, extra = self.validate(df)
        ok = np.array([len(h) == 0 for h in hard])
        score = np.full(len(d), np.nan)
        if ok.any():
            score[ok] = self.model.predict_proba(d.loc[ok, self.required])[:, 1]
        status = ["REJECTED: " + "; ".join(h) if h else ("SCORED_WITH_WARNINGS: " + "; ".join(s) if s else "SCORED") for h, s in zip(hard, soft)]
        if extra:
            status = [s if s.startswith("REJECTED") else (s + (" | " if s != "SCORED" else ": ") + f"ignored columns {extra}") if s != "SCORED" else f"SCORED_WITH_WARNINGS: ignored columns {extra}" for s in status]
        flag = pd.array([pd.NA if np.isnan(s) else int(s >= self.schema["threshold_top20"]) for s in score], dtype="Int64")
        return pd.DataFrame({"risk_score": score, "flag_top20": flag, "status": status, "model_status": self.MODEL_STATUS}, index=df.index)
