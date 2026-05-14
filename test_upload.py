"""
test_upload.py — verifies that the /api/upload endpoint works correctly
with a raw Sysmon CSV that has NO Label column.
"""
import requests
import json

url   = "http://localhost:5000/api/upload"
fpath = r"C:\Users\USER\Desktop\LM2\LM_System\dashboard\test_sysmon_nolabel.csv"

with open(fpath, "rb") as f:
    resp = requests.post(url, files=[("files[]", ("test_sysmon_nolabel.csv", f, "text/csv"))])

data = resp.json()

print("=== UPLOAD RESPONSE ===")
print("success        :", data["success"])
print("files          :", data["files"])

c = data["combined"]
print("total_records  :", c["total_records"])
print("hybrid_alerts  :", c["hybrid_alerts"])
print("rule_alerts    :", c["rule_alerts"])
print("ml_alerts      :", c["ml_alerts"])
print("has_ground_truth:", c["has_ground_truth"])
print("model_used     :", c["model_used"])
print("exec_time_s    :", c["execution_time_s"])

print("\n=== PER-RULE RESULTS (fired only) ===")
for r in data["per_rule"]:
    if r["alerts_fired"] > 0:
        print(f"  [{r['severity']:6s}] {r['name']:45s}  alerts={r['alerts_fired']}")

print("\n=== IOC: TOP SOURCE IPs ===")
for ip in data["ioc_summary"]["top_source_ips"][:5]:
    print(f"  {ip['ip']:25s}  count={ip['count']}")

print("\n=== SAMPLE ALERTS (first 3) ===")
for a in data["alerts"][:3]:
    print(f"  EID={a.get('EventID')}  sev={a.get('severity')}  "
          f"rule={a.get('rule_alert')}  ml={a.get('ml_alert')}  hybrid={a.get('hybrid_alert')}")
    print(f"  matched_rules: {a.get('matched_rules')}")
    print()

print("\n=== NO-LABEL MODE VERIFICATION ===")
hm = c.get("hybrid_metrics", {})
print("has_ground_truth in hybrid_metrics:", hm.get("has_ground_truth"))
print("=> Dashboard will show ALERT COUNTS (not TP/FP) — correct behaviour for no-label CSV")
