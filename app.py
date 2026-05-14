"""
dashboard/app.py — Flask API Server for the SOC Dashboard
==========================================================

Supports:
  - Serving the full 10-section SOC dashboard HTML
  - REST API for pre-computed metrics (fallback / Mission4 output)
  - CSV file upload endpoint that runs the hybrid detection pipeline live
    (Rule Engine + Isolation Forest) on any compatible Sysmon log CSV,
    then returns enriched per-row results and aggregate stats instantly.

Usage (run from the LM_System project root):
    python dashboard/app.py

Then open your browser at:
    http://localhost:5000
"""

import os
import sys
import io
import json
import uuid
import gc
import time
import traceback
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, send_file, jsonify, request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "mission4_hybrid_summary.json")
HTML_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
MODEL_PATH   = os.path.join(OUTPUTS_DIR, "isolation_forest_model.pkl")
THRESH_PATH  = os.path.join(OUTPUTS_DIR, "isolation_forest_threshold.pkl")

# Add the project root to sys.path so src.* imports work
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.isolation_forest import (
    encode_features,
    load_model,
    CATEGORICAL_OHE_FEATURES,
    NUMERIC_MINMAX_FEATURES,
)
from src.rule_engine import apply_rules

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB cap per upload

# ── Load trained model once at startup ──────────────────────────────────────
_model     = None
_threshold = None
_preprocessor = None

HIGH_CONFIDENCE_EVENTIDS = frozenset({7, 10, 23, 17})
THRESHOLD_DELTA = 0.02
NOISY_RULE_NAMES = (
    "LDAP AD Reconnaissance",
    "RPC/DCOM/WMI Remote Execution",
    "SMB / Windows Admin Shares Lateral Movement",
)

def _get_model():
    global _model, _threshold, _preprocessor
    if _model is None and os.path.exists(MODEL_PATH) and os.path.exists(THRESH_PATH):
        _model, _threshold = load_model()
        try:
            with open(MODEL_PATH, "rb") as f:
                pkg = pickle.load(f)
            if isinstance(pkg, dict):
                _preprocessor = pkg.get("preprocessor")
        except Exception:
            _preprocessor = None
    return _model, _threshold


def _get_final_if_threshold(base_threshold):
    """Use the latest Mission4 operating threshold for live dashboard uploads."""
    try:
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if summary.get("threshold_strict") is not None:
            return float(summary["threshold_strict"])
    except Exception:
        pass
    return float(base_threshold) - THRESHOLD_DELTA if base_threshold is not None else None


# ── Detection rule masks (vectorized, no external import needed) ───────────
def _apply_rules_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the 8 MITRE ATT&CK detection rules using vectorized pandas masks.
    Works on any DataFrame that has the expected Sysmon column names.
    Returns the DataFrame with added columns:
      rule_alert (bool), matched_rules (str), severity (str)
    """
    return apply_rules(df)

    df = df.copy()

    eid  = df.get("EventID", pd.Series(dtype=int))
    port = df.get("DestinationPortName", pd.Series(dtype=str)).astype(str).str.lower().str.strip()
    init = df.get("Initiated", pd.Series(dtype=str)).astype(str).str.lower().str.strip()
    src  = df.get("SourceIp",  pd.Series(dtype=str)).astype(str)

    def _is_internal(s: pd.Series) -> pd.Series:
        return (
            s.str.startswith("192.168.") | s.str.startswith("10.")
            | s.str.startswith("172.")   | s.str.startswith("fe80:")
        )

    def _initiated(i: pd.Series) -> pd.Series:
        return i.isin(["true", "1", "yes"])

    RULES = [
        # (name, mitre_id, severity, mask_expr)
        ("Malicious DLL / Module Load",        "T1574.001", "HIGH",
            eid == 7),
        ("LSASS / Cross-Process Handle Access","T1003.001", "HIGH",
            eid == 10),
        ("File Deletion / Evidence Removal",   "T1070.004", "HIGH",
            eid == 23),
        ("LDAP AD Reconnaissance",             "T1018",     "MEDIUM",
            (eid == 3) & port.isin(["ldap","389","msft-gc","3268"]) & _initiated(init) & _is_internal(src)),
        ("RPC/DCOM/WMI Remote Execution",      "T1021.003", "HIGH",
            (eid == 3) & port.isin(["epmap","135","msrpc"]) & _initiated(init) & _is_internal(src)),
        ("RDP Lateral Movement",               "T1021.001", "HIGH",
            (eid == 3) & port.isin(["ms-wbt-server","3389","rdp"]) & _initiated(init) & _is_internal(src)),
        ("Kerberos Ticket Abuse",              "T1558",     "HIGH",
            eid.isin([3,22]) & port.isin(["kerberos","88","kpasswd","464"]) & _initiated(init) & _is_internal(src)),
        ("Named Pipe Creation (C2 Channel)",   "T1559.001", "HIGH",
            eid == 17),
    ]

    SEVERITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    n = len(df)

    rule_alert_mask   = pd.Series(False, index=df.index)
    matched_rules_lst = [[] for _ in range(n)]
    severity_arr      = ["none"] * n

    for (name, _mid, sev, mask) in RULES:
        # Ensure mask aligns with df index
        try:
            aligned = mask.reindex(df.index, fill_value=False).astype(bool)
        except Exception:
            aligned = mask.astype(bool)

        rule_alert_mask = rule_alert_mask | aligned
        fired_idx = aligned[aligned].index.tolist()
        for idx_val in fired_idx:
            pos = df.index.get_loc(idx_val)
            matched_rules_lst[pos].append(name)
            if SEVERITY_RANK.get(sev, 0) > SEVERITY_RANK.get(severity_arr[pos], 0):
                severity_arr[pos] = sev

    df["rule_alert"]    = rule_alert_mask
    df["matched_rules"] = [", ".join(r) if r else "none" for r in matched_rules_lst]
    df["severity"]      = severity_arr
    return df


# ── Exact feature encoding — matches isolation_forest.py encode_features() ──
#
# The model was trained with:
#   - LabelEncoder on DestinationPortName and Computer (NOT OHE)
#   - SourceIp classified into 4 types (0=none, 1=internal, 2=external, 3=loopback)
#   - Initiated → encoded as 0/1/2
#   - 4 numeric cols: EventID, EventRecordID, Execution_ProcessID, ProcessId
#   - 5 engineered features: is_high_risk_event, event_rarity_score,
#                            same_process_ids, is_lm_port, is_internal_network_event
#   - NO MinMax scaling was applied to the training data
#
# This encoding is completely self-contained and works on ANY raw Sysmon CSV.

BENIGN_COUNT_PER_EVENTID = {
    1:  19530, 2:    234, 3: 1380345, 4:    439, 5:  172473,
    6:     24, 7:      2, 8:    330,  10:     1, 11:   9212,
    12:   661, 13: 17751, 15:   337,  16:   339, 17:     1,
    18:     1, 22:   9929, 23:     1,
}
TOTAL_BENIGN     = 1_611_619
HIGH_RISK_AIDS   = {7, 8, 10, 17, 18, 23}
LM_PORTS         = {"ldap","kerberos","epmap","ms-wbt-server","microsoft-ds","smb","rdp"}

EXPECTED_FEATURES = [
    "EventID", "EventRecordID", "Execution_ProcessID", "ProcessId",
    "DestinationPortName", "Initiated", "SourceIp_type", "Computer",
    "is_high_risk_event", "event_rarity_score", "same_process_ids",
    "is_lm_port", "is_internal_network_event",
]


def _encode_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the exact encode_features() logic from src/isolation_forest.py.
    Produces the same 13-column numeric matrix the model was trained on.
    Works on raw (unpreprocessed) Sysmon CSV files with no scaler needed.
    """
    enc = pd.DataFrame(index=df.index)

    # ── 1. Numeric columns ─────────────────────────────────────────────────
    for col in ["EventID", "EventRecordID", "Execution_ProcessID", "ProcessId"]:
        enc[col] = pd.to_numeric(df.get(col, pd.Series(0, index=df.index)),
                                 errors="coerce").fillna(0)

    # ── 2. DestinationPortName — label-encode to integer ──────────────────
    port_raw = df.get("DestinationPortName",
                      pd.Series("unknown", index=df.index)).astype(str).str.lower().str.strip()
    # Map each unique string to a stable integer
    port_cats = {p: i for i, p in enumerate(sorted(port_raw.unique()))}
    enc["DestinationPortName"] = port_raw.map(port_cats).fillna(0).astype(int)

    # ── 3. Initiated → 0 (inbound) / 1 (outbound) / 2 (local) ───────────
    lower = df.get("Initiated",
                   pd.Series("0", index=df.index)).astype(str).str.lower().str.strip()
    init_enc = pd.Series(2, index=df.index, dtype=int)
    init_enc[lower.isin(["true", "1", "yes"])] = 1
    init_enc[lower == "false"]                 = 0
    enc["Initiated"] = init_enc

    # ── 4. SourceIp → classify into 4 IP types ───────────────────────────
    src = df.get("SourceIp", pd.Series("0", index=df.index)).astype(str).str.strip()
    ip_type = pd.Series(2, index=df.index, dtype=int)   # default: external
    ip_type[src == "0"]               = 0               # no IP (local event)
    ip_type[src == "127.0.0.1"]       = 3               # loopback
    internal_mask = (
        src.str.startswith("192.168.") | src.str.startswith("10.")
        | src.str.startswith("172.")   | src.str.startswith("fe80:")
        | src.str.startswith("0:0:0:0:0:0:0:1")
    )
    ip_type[internal_mask] = 1
    enc["SourceIp_type"] = ip_type

    # ── 5. Computer — label-encode hostname ───────────────────────────────
    comp_raw = df.get("Computer", pd.Series("unknown", index=df.index)).astype(str).str.strip()
    comp_cats = {c: i for i, c in enumerate(sorted(comp_raw.unique()))}
    enc["Computer"] = comp_raw.map(comp_cats).fillna(0).astype(int)

    # ── 6. Engineered: is_high_risk_event ────────────────────────────────
    enc["is_high_risk_event"] = enc["EventID"].isin(HIGH_RISK_AIDS).astype(int)

    # ── 7. Engineered: event_rarity_score ────────────────────────────────
    benign_frac = enc["EventID"].map(
        {eid: cnt / TOTAL_BENIGN for eid, cnt in BENIGN_COUNT_PER_EVENTID.items()}
    ).fillna(1e-7)
    enc["event_rarity_score"] = -np.log10(benign_frac + 1e-7)

    # ── 8. Engineered: same_process_ids ──────────────────────────────────
    enc["same_process_ids"] = (
        (enc["Execution_ProcessID"] != 0)
        & (enc["ProcessId"] != 0)
        & (enc["Execution_ProcessID"] == enc["ProcessId"])
    ).astype(int)

    # ── 9. Engineered: is_lm_port ────────────────────────────────────────
    enc["is_lm_port"] = port_raw.isin(LM_PORTS).astype(int)

    # ── 10. Engineered: is_internal_network_event ─────────────────────────
    enc["is_internal_network_event"] = (
        (enc["SourceIp_type"] == 1) & (enc["EventID"] == 3)
    ).astype(int)

    return enc.fillna(0).astype(np.float32)


def _apply_ml_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the trained Isolation Forest on uploaded data.
    Uses the exact same feature encoding as the training pipeline.
    Works on raw (unpreprocessed) Sysmon CSV — no scaler needed.
    Adds ml_score (float) and ml_alert (bool) columns.
    """
    try:
        model, threshold = _get_model()
    except Exception as e:
        print(f"  [ML] Model load failed: {e}")
        df["ml_score"] = 0.0
        df["ml_alert"] = False
        df["ml_alert_strict"] = False
        df.attrs["model_used"] = False
        return df

    if model is None:
        df["ml_score"] = 0.0
        df["ml_alert"] = False
        df["ml_alert_strict"] = False
        df.attrs["model_used"] = False
        return df

    try:
        for col in CATEGORICAL_OHE_FEATURES:
            if col not in df.columns:
                df[col] = "missing"
        for col in NUMERIC_MINMAX_FEATURES:
            if col not in df.columns:
                df[col] = 0

        if _preprocessor is not None:
            work = df[CATEGORICAL_OHE_FEATURES + NUMERIC_MINMAX_FEATURES].copy()
            for col in CATEGORICAL_OHE_FEATURES:
                work[col] = work[col].astype(str).fillna("missing")
            for col in NUMERIC_MINMAX_FEATURES:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
            feat = _preprocessor.transform(work)
        else:
            feat = encode_features(df)
        scores = model.decision_function(feat)
        df["ml_score"] = scores

        if threshold is not None:
            base_threshold = float(threshold)
            strict_threshold = _get_final_if_threshold(base_threshold)
            df["ml_alert"] = scores < base_threshold
            df["ml_alert_strict"] = scores < strict_threshold
        else:
            fallback_threshold = np.percentile(scores, 10)
            df["ml_alert"] = scores < fallback_threshold
            df["ml_alert_strict"] = df["ml_alert"]
        df.attrs["model_used"] = True
        return df

        # Align columns to what the model expects (in case feature list differs)
        try:
            expected = list(model.feature_names_in_)
            for c in expected:
                if c not in feat.columns:
                    feat[c] = 0.0
            feat = feat[expected]
        except AttributeError:
            # Older sklearn — align to our known feature order
            for c in EXPECTED_FEATURES:
                if c not in feat.columns:
                    feat[c] = 0.0
            feat = feat[EXPECTED_FEATURES]

        scores = model.decision_function(feat)
        df["ml_score"] = scores

        if threshold is not None:
            base_threshold = float(threshold)
            strict_threshold = _get_final_if_threshold(base_threshold)
            df["ml_alert"] = scores < base_threshold
            df["ml_alert_strict"] = scores < strict_threshold
        else:
            # Fallback: use 10th percentile of this batch's scores
            fallback_threshold = np.percentile(scores, 10)
            df["ml_alert"] = scores < fallback_threshold
            df["ml_alert_strict"] = df["ml_alert"]
        df.attrs["model_used"] = True

    except Exception as e:
        print(f"  [ML] Scoring failed: {e}")
        df["ml_score"] = 0.0
        df["ml_alert"] = False
        df["ml_alert_strict"] = False
        df.attrs["model_used"] = False

    return df



# ── Fallback data ─────────────────────────────────────────────────────────────
FALLBACK_DATA = {
    "_fallback": True,
    "meta": {
        "title":             "LMD-2023 Hybrid Lateral Movement Detection System",
        "dataset":           "Smiliotopoulos et al. (2025): LMD-2023",
        "generated_at":      None,
        "total_records":     1752836,
        "malicious_records": 141217,
        "benign_records":    1611619,
        "attack_rate_pct":   8.06,
        "execution_time_s":  None,
    },
    "rules": {
        "TP": 103933, "FP": 2446, "TN": 1609173, "FN": 37284,
        "detection_rate_pct":      73.5981,
        "false_positive_rate_pct":  0.1518,
        "precision_pct":           97.7007,
        "f1_pct":                  83.9537,
        "accuracy_pct":            97.7334,
    },
    "ml": {
        "TP": 136159, "FP": 47092, "TN": 1564527, "FN": 5058,
        "detection_rate_pct":      96.4183,
        "false_positive_rate_pct":  2.9220,
        "precision_pct":           74.3019,
        "f1_pct":                  83.9275,
        "accuracy_pct":            97.0180,
    },
    "hybrid": {
        "TP": 136159, "FP": 47092, "TN": 1564527, "FN": 5058,
        "detection_rate_pct":      96.4183,
        "false_positive_rate_pct":  2.9220,
        "precision_pct":           74.3019,
        "f1_pct":                  83.9275,
        "accuracy_pct":            97.0180,
    },
    "fn_recovered_by_ml": 32226,
    "per_rule": [
        {"name": "Malicious DLL / Module Load",        "mitre_id": "T1574.001", "severity": "HIGH",
         "alerts_fired": 46869, "tp": 46867, "fp": 2,    "precision_pct": 100.0, "recall_pct": 33.2,
         "description": "Detects Sysmon Event 7 (ImageLoad). 99.99% of Event 7 records are attack events."},
        {"name": "LSASS / Cross-Process Handle Access","mitre_id": "T1003.001", "severity": "HIGH",
         "alerts_fired": 32176, "tp": 32175, "fp": 1,    "precision_pct": 100.0, "recall_pct": 22.8,
         "description": "Detects Sysmon Event 10 (ProcessAccess). Used by Mimikatz and credential dumping tools."},
        {"name": "File Deletion / Evidence Removal",   "mitre_id": "T1070.004", "severity": "HIGH",
         "alerts_fired": 22764, "tp": 22763, "fp": 1,    "precision_pct": 100.0, "recall_pct": 16.1,
         "description": "Detects Sysmon Event 23 (FileDelete). Attackers delete tools and logs post-exploitation."},
        {"name": "LDAP AD Reconnaissance",             "mitre_id": "T1018",     "severity": "MEDIUM",
         "alerts_fired": 3256,  "tp":  1560, "fp": 1696, "precision_pct":  47.9, "recall_pct":  1.1,
         "description": "Detects Event 3 to LDAP port from internal hosts. Indicates BloodHound/PowerView enumeration."},
        {"name": "RPC/DCOM/WMI Remote Execution",      "mitre_id": "T1021.003", "severity": "HIGH",
         "alerts_fired": 1245,  "tp":   505, "fp":  740, "precision_pct":  40.6, "recall_pct":  0.4,
         "description": "Detects Event 3 to RPC Endpoint Mapper (port 135). Used by PsExec and WMI lateral movement."},
        {"name": "RDP Lateral Movement",               "mitre_id": "T1021.001", "severity": "HIGH",
         "alerts_fired":   42,  "tp":    37, "fp":    5, "precision_pct":  88.1, "recall_pct":  0.0,
         "description": "Detects outbound RDP (port 3389) from internal hosts. Most direct lateral movement technique."},
        {"name": "Kerberos Ticket Abuse",              "mitre_id": "T1558",     "severity": "HIGH",
         "alerts_fired":   12,  "tp":    12, "fp":    0, "precision_pct": 100.0, "recall_pct":  0.0,
         "description": "Detects Kerberos port (88) activity. Indicates Pass-the-Ticket or Kerberoasting attacks."},
        {"name": "Named Pipe Creation (C2 Channel)",   "mitre_id": "T1559.001", "severity": "HIGH",
         "alerts_fired":   15,  "tp":    14, "fp":    1, "precision_pct":  93.3, "recall_pct":  0.0,
         "description": "Detects Sysmon Event 17 (Pipe Created). Used by Cobalt Strike SMB Beacon as C2 channel."},
    ],
}


def load_summary() -> dict:
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return FALLBACK_DATA


# ── Routes — static / pre-computed data ───────────────────────────────────────

@app.route("/")
def index():
    return send_file(HTML_PATH)


@app.route("/api/metrics")
def api_metrics():
    data = load_summary()
    return jsonify({
        "meta":               data.get("meta", {}),
        "rules":              data.get("rules", {}),
        "ml":                 data.get("ml", {}),
        "hybrid":             data.get("hybrid", {}),
        "fn_recovered_by_ml": data.get("fn_recovered_by_ml", 0),
        "is_fallback":        data.get("_fallback", False),
    })


@app.route("/api/rules")
def api_rules():
    data = load_summary()
    return jsonify(data.get("per_rule", []))


@app.route("/api/status")
def api_status():
    summary_exists = os.path.exists(SUMMARY_PATH)
    model_exists   = os.path.exists(MODEL_PATH)
    data           = load_summary()
    return jsonify({
        "mission4_complete": summary_exists,
        "model_trained":     model_exists,
        "using_fallback":    data.get("_fallback", False),
        "generated_at":      data.get("meta", {}).get("generated_at"),
        "server_time":       datetime.now().isoformat(),
    })


# ── Upload endpoint ────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Accept one or more CSV files, run the hybrid detection pipeline,
    and return aggregated results + per-row alert data.

    Expects multipart/form-data with field name 'files[]' (or 'file').
    Returns JSON:
      {
        "success": true,
        "files": [...],          # per-file summaries
        "combined": {...},       # aggregate stats across all files
        "alerts": [...],         # top-N alert rows (capped for response size)
        "ioc_summary": {...},
        "timeline": [...],
        "per_rule": [...],
      }
    """
    t0 = time.time()

    # Accept either 'files[]' (batch) or 'file' (single)
    uploaded = request.files.getlist("files[]") or request.files.getlist("file")
    if not uploaded:
        return jsonify({"success": False, "error": "No files received."}), 400

    all_dfs      = []
    file_summaries = []
    parse_errors   = []

    for f in uploaded:
        if not f or not f.filename:
            continue
        filename = f.filename
        try:
            raw_bytes = f.read()
            # Try comma first, then semicolon/tab
            for sep in [",", ";", "\t"]:
                try:
                    chunk = pd.read_csv(
                        io.BytesIO(raw_bytes), sep=sep,
                        low_memory=False, nrows=500_000  # safety cap per file
                    )
                    if len(chunk.columns) > 3:
                        break
                except Exception:
                    continue
            else:
                raise ValueError("Cannot parse CSV — tried , ; and tab separators.")

            n_rows = len(chunk)
            file_summaries.append({"filename": filename, "rows": n_rows, "status": "ok"})
            all_dfs.append(chunk)
        except Exception as e:
            parse_errors.append({"filename": filename, "error": str(e)})

    if not all_dfs:
        return jsonify({
            "success": False,
            "error": "No valid CSV data found.",
            "parse_errors": parse_errors,
        }), 422

    # Merge all uploaded files
    df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    n_total = len(df)

    # ── Run rule engine ──────────────────────────────────────────────────
    df = _apply_rules_fast(df)

    # ── Run ML model (if available) ──────────────────────────────────────
    df = _apply_ml_fast(df)
    model_available = bool(df.attrs.get("model_used", False))

    # ── Hybrid decision: same precision-preserving fusion as Mission 4 ───
    event_ids = pd.to_numeric(df.get("EventID", pd.Series(0, index=df.index)),
                              errors="coerce").fillna(0).astype(int)
    rule_alert = df["rule_alert"].astype(bool)
    ml_strict = df.get("ml_alert_strict", df["ml_alert"]).astype(bool)
    matched_rules = df.get("matched_rules", pd.Series("", index=df.index)).fillna("")
    noisy_rule_alert = pd.Series(False, index=df.index)
    for rule_name in NOISY_RULE_NAMES:
        noisy_rule_alert |= matched_rules.str.contains(rule_name, regex=False)
    trusted_rule_alert = rule_alert & ~noisy_rule_alert
    rule_blind_spot = ~rule_alert & ~event_ids.isin(HIGH_CONFIDENCE_EVENTIDS)
    ml_candidate_zone = rule_blind_spot | noisy_rule_alert
    df["hybrid_alert"] = trusted_rule_alert | (ml_strict & ml_candidate_zone)

    # ── Compute metrics (with label if present) ──────────────────────────
    has_label      = "Label" in df.columns
    label_col      = None
    if has_label:
        # Normalise label: 1/True/"Attack"/"1"/"attack" → True
        raw_label = df["Label"].astype(str).str.lower().str.strip()
        label_col = (
            raw_label.isin(["1", "attack", "true", "malicious", "malware", "yes"])
        ).astype(bool)

    def _metrics(pred: pd.Series, truth=label_col) -> dict:
        if truth is None:
            return {"has_ground_truth": False}
        tp = int((pred  & truth).sum())
        fp = int((pred  & ~truth).sum())
        tn = int((~pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        total_att = tp + fn
        total_ben = fp + tn
        dr  = tp / total_att * 100 if total_att > 0 else 0
        fpr = fp / total_ben * 100 if total_ben > 0 else 0
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        f1   = (2 * prec * dr) / (prec + dr) if (prec + dr) > 0 else 0
        acc  = (tp + tn) / (tp + fp + tn + fn) * 100
        return {
            "has_ground_truth": True,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "detection_rate_pct":      round(dr,  4),
            "false_positive_rate_pct": round(fpr, 4),
            "precision_pct":           round(prec,4),
            "f1_pct":                  round(f1,  4),
            "accuracy_pct":            round(acc, 4),
            "total_malicious":         total_att,
            "total_benign":            total_ben,
        }

    rule_metrics   = _metrics(df["rule_alert"])
    ml_metrics     = _metrics(df["ml_alert"])
    hybrid_metrics = _metrics(df["hybrid_alert"])

    # alert counts (when no ground truth)
    n_rule_alerts   = int(df["rule_alert"].sum())
    n_ml_alerts     = int(df["ml_alert"].sum())
    n_hybrid_alerts = int(df["hybrid_alert"].sum())
    n_rule_only_final = int((df["hybrid_alert"] & df["rule_alert"] & ~df["ml_alert_strict"]).sum())
    n_ml_only_final   = int((df["hybrid_alert"] & ~df["rule_alert"] & df["ml_alert_strict"]).sum())
    n_both_final      = int((df["hybrid_alert"] & df["rule_alert"] & df["ml_alert_strict"]).sum())

    # ── Per-rule breakdown ───────────────────────────────────────────────
    RULE_DEF = [
        ("Malicious DLL / Module Load",        "T1574.001", "HIGH"),
        ("LSASS / Cross-Process Handle Access","T1003.001", "HIGH"),
        ("File Deletion / Evidence Removal",   "T1070.004", "HIGH"),
        ("LDAP AD Reconnaissance",             "T1018",     "MEDIUM"),
        ("RPC/DCOM/WMI Remote Execution",      "T1021.003", "HIGH"),
        ("RDP Lateral Movement",               "T1021.001", "HIGH"),
        ("Kerberos Ticket Abuse",              "T1558",     "HIGH"),
        ("Named Pipe Creation (C2 Channel)",   "T1559.001", "HIGH"),
    ]
    per_rule_results = []
    for (name, mid, sev) in RULE_DEF:
        fired = df["matched_rules"].str.contains(name, regex=False, na=False)
        alerts_fired = int(fired.sum())
        if has_label and label_col is not None:
            tp = int((fired & label_col).sum())
            fp = int((fired & ~label_col).sum())
        else:
            tp, fp = alerts_fired, 0
        prec_  = tp / alerts_fired * 100 if alerts_fired > 0 else 0
        per_rule_results.append({
            "name": name, "mitre_id": mid, "severity": sev,
            "alerts_fired": alerts_fired,
            "tp": tp, "fp": fp,
            "precision_pct": round(prec_, 2),
            "recall_pct": 0,
        })

    # ── IOC summary ──────────────────────────────────────────────────────
    ioc_summary = {}
    if "SourceIp" in df.columns:
        alerted = df[df["hybrid_alert"]]
        top_ips = alerted["SourceIp"].value_counts().head(20).to_dict()
        ioc_summary["top_source_ips"] = [{"ip": k, "count": v} for k, v in top_ips.items()]
    else:
        ioc_summary["top_source_ips"] = []

    if "DestinationPortName" in df.columns:
        top_ports = df[df["hybrid_alert"]]["DestinationPortName"].value_counts().head(10).to_dict()
        ioc_summary["top_ports"] = [{"port": k, "count": v} for k, v in top_ports.items()]
    else:
        ioc_summary["top_ports"] = []

    if "Computer" in df.columns:
        top_hosts = df[df["hybrid_alert"]]["Computer"].value_counts().head(10).to_dict()
        ioc_summary["top_hosts"] = [{"host": k, "count": v} for k, v in top_hosts.items()]
    else:
        ioc_summary["top_hosts"] = []

    # ── Timeline: event counts by EventID ───────────────────────────────
    timeline = []
    if "EventID" in df.columns:
        eid_counts = df["EventID"].value_counts().head(20).to_dict()
        timeline = [{"event_id": k, "total": v,
                     "alerted": int(df[(df["EventID"] == k) & df["hybrid_alert"]].shape[0])}
                    for k, v in eid_counts.items()]

    # ── Alert rows (top 200 for dashboard table) ─────────────────────────
    alert_df = df[df["hybrid_alert"]].copy()
    # select presentable columns
    keep_cols = [c for c in [
        "EventID", "Computer", "SourceIp", "DestinationPortName",
        "Initiated", "ProcessId", "Label",
        "rule_alert", "ml_alert", "ml_alert_strict", "hybrid_alert",
        "matched_rules", "severity", "ml_score",
    ] if c in alert_df.columns]
    alert_rows = alert_df[keep_cols].head(200).replace({np.nan: None}).to_dict(orient="records")

    # ── Lateral movement graph (source→dest pairs) ───────────────────────
    lm_edges = []
    if "SourceIp" in df.columns and "Computer" in df.columns:
        edge_counts = (
            df[df["hybrid_alert"]]
            .groupby(["SourceIp", "Computer"])
            .size()
            .reset_index(name="weight")
            .sort_values("weight", ascending=False)
            .head(50)
        )
        lm_edges = edge_counts.to_dict(orient="records")

    exec_time = round(time.time() - t0, 2)

    combined = {
        "total_records":    n_total,
        "rule_alerts":      n_rule_alerts,
        "ml_alerts":        n_ml_alerts,
        "hybrid_alerts":    n_hybrid_alerts,
        "rule_only_final":  n_rule_only_final,
        "ml_only_final":    n_ml_only_final,
        "both_final":       n_both_final,
        "model_used":       model_available,
        "has_ground_truth": has_label,
        "rule_metrics":     rule_metrics,
        "ml_metrics":       ml_metrics,
        "hybrid_metrics":   hybrid_metrics,
        "execution_time_s": exec_time,
        "processed_at":     datetime.now().isoformat(),
    }

    return jsonify({
        "success":     True,
        "files":       file_summaries,
        "parse_errors":parse_errors,
        "combined":    combined,
        "alerts":      alert_rows,
        "ioc_summary": ioc_summary,
        "timeline":    timeline,
        "per_rule":    per_rule_results,
        "lm_edges":    lm_edges,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LMD-SOC SENTINEL - Dashboard Server")
    print("=" * 60)
    print(f"  Mission 4 data : {'FOUND [OK]' if os.path.exists(SUMMARY_PATH) else 'NOT FOUND  (using fallback data)'}")
    print(f"  ML Model       : {'LOADED [OK]' if os.path.exists(MODEL_PATH) else 'NOT FOUND  (rule-only mode)'}")
    print(f"  Dashboard URL  : http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
