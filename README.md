# AI-Powered Lightweight Early Lateral Movement Detection System

This project implements a two-layer lateral movement detection pipeline for Windows Sysmon telemetry:

1. A rule-based detection layer aligned with MITRE ATT&CK.
2. An Isolation Forest anomaly-detection layer.

The final system combines both layers in a hybrid pipeline and includes a local SOC-style dashboard for interactive testing and CSV upload analysis.

## Main Files

- `Mission1_run.py` - data acquisition and preprocessing
- `Mission2_run.py` - rule-based detection engine
- `Mission3_run.py` - Isolation Forest model training and evaluation
- `Mission4_run.py` - final hybrid detection pipeline

## Core Modules

- `src/config.py`
- `src/data_loader.py`
- `src/preprocessing.py`
- `src/rule_engine.py`
- `src/rule_evaluator.py`
- `src/isolation_forest.py`

## Dashboard

- `dashboard/app.py`
- `dashboard/index.html`

Included sample upload files:

- `dashboard/family2_test_upload_unlabeled.csv`
- `dashboard/random_soc_test_unlabeled.csv`

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- The large original LMD-2023 dataset is not included in this repository.
- Generated outputs, trained model files, and temporary experiment artifacts are intentionally excluded.
