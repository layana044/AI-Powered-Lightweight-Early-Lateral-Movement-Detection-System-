# ============================================================
# Lateral Movement Detection System
# Step 5: Rule-Based Correlation Engine
#
# Purpose:
# Detect known lateral movement patterns by analyzing
# Sysmon events within record-based time windows.
#
# Design Notes:
# - Events are analyzed per host (Computer).
# - EventRecordID is used as a proxy for time.
# - Rules 2 and 7 are mutually exclusive.
# - Rules 3 and 6 are mutually exclusive.
# - Rule 8 requires strict sequential ordering.
# ============================================================

import sys
from datetime import datetime

import pandas as pd
import config


# ============================================================
# Step 1: Load Dataset
# ============================================================

print("=" * 60)
print("      LATERAL MOVEMENT RULE-BASED DETECTION ENGINE     ")
print("=" * 60)

print("\n" + "=" * 60)
print("  STEP 1: LOADING DATASET                              ")
print("=" * 60)

dataset = pd.read_csv(config.DATASET_FILE_PATH)

print(f"Dataset file  : {config.DATASET_FILE_PATH}")
print(f"Total rows    : {len(dataset)}")
print(f"Total columns : {dataset.shape[1]}")
print(f"Columns       : {dataset.columns.tolist()}")


# ============================================================
# Step 2: Validate Required Columns
# ============================================================

print("\n" + "=" * 60)
print("  STEP 2: VALIDATING REQUIRED COLUMNS                 ")
print("=" * 60)

all_columns_present = True

for column_name in config.REQUIRED_COLUMNS:
    if column_name in dataset.columns:
        print(f"FOUND   : {column_name}")
    else:
        print(f"MISSING : {column_name}")
        all_columns_present = False

if not all_columns_present:
    print("\nERROR: Some required columns are missing.")
    sys.exit(1)

print("\nAll required columns are present.")


# ============================================================
# Step 3: Basic Data Cleaning / Type Preparation
# ============================================================

print("\n" + "=" * 60)
print("  STEP 3: PREPARING DATA                              ")
print("=" * 60)

dataset = dataset.copy()

dataset['EventID'] = pd.to_numeric(dataset['EventID'], errors='coerce')
dataset['EventRecordID'] = pd.to_numeric(dataset['EventRecordID'], errors='coerce')
dataset['Label'] = pd.to_numeric(dataset['Label'], errors='coerce')

dataset = dataset.dropna(
    subset=['Computer', 'EventID', 'EventRecordID', 'Label']
)

dataset['EventID'] = dataset['EventID'].astype(int)
dataset['EventRecordID'] = dataset['EventRecordID'].astype(int)
dataset['Label'] = dataset['Label'].astype(int)

print(f"Rows after cleaning: {len(dataset)}")


# ============================================================
# Helper Functions
# ============================================================

def create_detection_alert(
    alert_id,
    rule_number,
    rule_name,
    severity,
    source_host,
    description,
    mitre_technique,
    evidence_count,
    attack_label,
    matched_record_ids
):
    return {
        'Alert ID': alert_id,
        'Rule Number': rule_number,
        'Rule Name': rule_name,
        'Severity': severity,
        'Status': config.STATUS_UNRESOLVED,
        'Source Host': source_host,
        'Description': description,
        'MITRE Technique': mitre_technique,
        'Evidence Count': evidence_count,
        'Attack Label': attack_label,
        'Is Real Attack': 'YES' if attack_label == 1 else 'NO',
        'Matched Record IDs': ', '.join(map(str, matched_record_ids)),
        'Detection Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def check_events_in_window(record_id_list, window_size, threshold):
    """
    Returns:
        (True, matched_record_ids) if a burst is detected
        (False, []) otherwise

    Trigger condition:
        number of events inside the window > threshold
    """
    sorted_record_ids = sorted(record_id_list)

    left = 0
    for right in range(len(sorted_record_ids)):
        while sorted_record_ids[right] - sorted_record_ids[left] > window_size:
            left += 1

        current_window_ids = sorted_record_ids[left:right + 1]

        if len(current_window_ids) > threshold:
            return True, current_window_ids

    return False, []


def get_attack_label_for_host(computer_logs):
    return int(computer_logs['Label'].max())


# ============================================================
# Step 4: Initialize Alert Storage
# ============================================================

all_generated_alerts = []
current_alert_id = 1


# ============================================================
# Step 5: Group Logs by Computer
# ============================================================

print("\n" + "=" * 60)
print("  STEP 4: GROUPING LOGS BY COMPUTER                   ")
print("=" * 60)

logs_grouped_by_computer = dataset.groupby('Computer')
total_computers = len(logs_grouped_by_computer)

print(f"Total unique computers : {total_computers}")

print("\n" + "=" * 60)
print("  STEP 5: APPLYING DETECTION RULES                    ")
print("=" * 60)
print("Analyzing each computer... please wait\n")


# ============================================================
# Main Rule Evaluation Loop
# ============================================================

for computer_name, computer_logs in logs_grouped_by_computer:

    computer_logs = computer_logs.sort_values(
        by='EventRecordID'
    ).reset_index(drop=True)

    computer_attack_label = get_attack_label_for_host(computer_logs)

    # Mutual exclusion trackers
    ldap_flood_triggered = False
    rdp_rapid_exploit_triggered = False

    # --------------------------------------------------------
    # RULE 1: SMB Rapid Scan / Exploitation Pattern
    # --------------------------------------------------------
    smb_connection_events = computer_logs[
        (computer_logs['EventID'] == config.EVENTID_NETWORK_CONNECTION) &
        (computer_logs['DestinationPortName'] == config.PORT_SMB)
    ]

    if len(smb_connection_events) > config.RULE_1_SMB_RAPID_SCAN_THRESHOLD:
        smb_record_ids = smb_connection_events['EventRecordID'].tolist()

        rule_1_triggered, matched_ids = check_events_in_window(
            record_id_list=smb_record_ids,
            window_size=config.RULE_1_SMB_RAPID_SCAN_TIME_WINDOW,
            threshold=config.RULE_1_SMB_RAPID_SCAN_THRESHOLD
        )

        if rule_1_triggered:
            alert = create_detection_alert(
                alert_id=current_alert_id,
                rule_number=1,
                rule_name='SMB Rapid Scan / Exploitation Pattern',
                severity=config.SEVERITY_CRITICAL,
                source_host=computer_name,
                description=(
                    f'Computer [{computer_name}] generated more than '
                    f'{config.RULE_1_SMB_RAPID_SCAN_THRESHOLD} SMB '
                    f'connections within '
                    f'{config.RULE_1_SMB_RAPID_SCAN_TIME_WINDOW} '
                    f'sequential log records. This indicates rapid SMB-based '
                    f'scanning or exploitation behavior consistent with '
                    f'remote service exploitation over SMB.'
                ),
                mitre_technique=config.MITRE_EXPLOITATION_REMOTE_SERVICES,
                evidence_count=len(matched_ids),
                attack_label=computer_attack_label,
                matched_record_ids=matched_ids
            )

            all_generated_alerts.append(alert)
            current_alert_id += 1

    # --------------------------------------------------------
    # RULE 2: RDP Rapid Exploitation Pattern
    # --------------------------------------------------------
    rdp_connection_events = computer_logs[
        (computer_logs['EventID'] == config.EVENTID_NETWORK_CONNECTION) &
        (computer_logs['DestinationPortName'] == config.PORT_RDP)
    ]

    if len(rdp_connection_events) > config.RULE_2_RDP_RAPID_EXPLOIT_THRESHOLD:
        rdp_record_ids = rdp_connection_events['EventRecordID'].tolist()

        rule_2_triggered, matched_ids = check_events_in_window(
            record_id_list=rdp_record_ids,
            window_size=config.RULE_2_RDP_RAPID_EXPLOIT_TIME_WINDOW,
            threshold=config.RULE_2_RDP_RAPID_EXPLOIT_THRESHOLD
        )

        if rule_2_triggered:
            rdp_rapid_exploit_triggered = True

            alert = create_detection_alert(
                alert_id=current_alert_id,
                rule_number=2,
                rule_name='RDP Rapid Exploitation Pattern',
                severity=config.SEVERITY_CRITICAL,
                source_host=computer_name,
                description=(
                    f'Computer [{computer_name}] generated more than '
                    f'{config.RULE_2_RDP_RAPID_EXPLOIT_THRESHOLD} RDP '
                    f'connections within '
                    f'{config.RULE_2_RDP_RAPID_EXPLOIT_TIME_WINDOW} '
                    f'sequential log records. This indicates rapid RDP-based '
                    f'exploitation behavior consistent with remote service '
                    f'exploitation over RDP.'
                ),
                mitre_technique=config.MITRE_EXPLOITATION_REMOTE_SERVICES,
                evidence_count=len(matched_ids),
                attack_label=computer_attack_label,
                matched_record_ids=matched_ids
            )

            all_generated_alerts.append(alert)
            current_alert_id += 1

    # --------------------------------------------------------
    # RULE 3: LDAP Flood Pattern
    # --------------------------------------------------------
    ldap_connection_events = computer_logs[
        (computer_logs['EventID'] == config.EVENTID_NETWORK_CONNECTION) &
        (computer_logs['DestinationPortName'] == config.PORT_LDAP)
    ]

    if len(ldap_connection_events) > config.RULE_3_LDAP_FLOOD_THRESHOLD:
        ldap_record_ids = ldap_connection_events['EventRecordID'].tolist()

        rule_3_triggered, matched_ids = check_events_in_window(
            record_id_list=ldap_record_ids,
            window_size=config.RULE_3_LDAP_FLOOD_TIME_WINDOW,
            threshold=config.RULE_3_LDAP_FLOOD_THRESHOLD
        )

        if rule_3_triggered:
            ldap_flood_triggered = True

            alert = create_detection_alert(
                alert_id=current_alert_id,
                rule_number=3,
                rule_name='LDAP Flood Pattern',
                severity=config.SEVERITY_CRITICAL,
                source_host=computer_name,
                description=(
                    f'Computer [{computer_name}] generated more than '
                    f'{config.RULE_3_LDAP_FLOOD_THRESHOLD} LDAP '
                    f'connections within '
                    f'{config.RULE_3_LDAP_FLOOD_TIME_WINDOW} '
                    f'sequential log records. This indicates a dense LDAP '
                    f'flood pattern consistent with aggressive exploitation '
                    f'or authentication-bypass activity against directory '
                    f'services.'
                ),
                mitre_technique=config.MITRE_EXPLOITATION_REMOTE_SERVICES,
                evidence_count=len(matched_ids),
                attack_label=computer_attack_label,
                matched_record_ids=matched_ids
            )

            all_generated_alerts.append(alert)
            current_alert_id += 1

    # --------------------------------------------------------
    # RULE 4: Suspicious Process Access Activity
    # --------------------------------------------------------
    process_access_events = computer_logs[
        computer_logs['EventID'] == config.EVENTID_PROCESS_ACCESS
    ]

    if len(process_access_events) > config.RULE_4_PROCESS_ACCESS_THRESHOLD:
        matched_ids = process_access_events['EventRecordID'].tolist()

        alert = create_detection_alert(
            alert_id=current_alert_id,
            rule_number=4,
            rule_name='Suspicious Process Access Activity',
            severity=config.SEVERITY_CRITICAL,
            source_host=computer_name,
            description=(
                f'Computer [{computer_name}] generated '
                f'{len(process_access_events)} process access events '
                f'(EventID 10), exceeding the threshold of '
                f'{config.RULE_4_PROCESS_ACCESS_THRESHOLD}. This indicates '
                f'suspicious process memory access activity that may be '
                f'associated with credential dumping behavior.'
            ),
            mitre_technique=config.MITRE_OS_CREDENTIAL_DUMPING,
            evidence_count=len(matched_ids),
            attack_label=computer_attack_label,
            matched_record_ids=matched_ids
        )

        all_generated_alerts.append(alert)
        current_alert_id += 1

    # --------------------------------------------------------
    # RULE 5: Process Creation Followed by SMB Activity
    # --------------------------------------------------------
    process_creation_events = computer_logs[
        computer_logs['EventID'] == config.EVENTID_PROCESS_CREATION
    ]

    smb_follow_events = computer_logs[
        (computer_logs['EventID'] == config.EVENTID_NETWORK_CONNECTION) &
        (computer_logs['DestinationPortName'] == config.PORT_SMB)
    ]

    if len(process_creation_events) > 0 and len(smb_follow_events) > 0:
        rule_5_triggered = False
        matched_ids = []

        for _, process_event in process_creation_events.iterrows():
            process_record_id = process_event['EventRecordID']
            window_end = process_record_id + config.RULE_5_PROCESS_TO_SMB_TIME_WINDOW

            smb_events_after_process = smb_follow_events[
                (smb_follow_events['EventRecordID'] > process_record_id) &
                (smb_follow_events['EventRecordID'] <= window_end)
            ]

            if len(smb_events_after_process) > 0:
                rule_5_triggered = True
                matched_ids = [process_record_id] + smb_events_after_process['EventRecordID'].tolist()
                break

        if rule_5_triggered:
            alert = create_detection_alert(
                alert_id=current_alert_id,
                rule_number=5,
                rule_name='Process Creation Followed by SMB Activity',
                severity=config.SEVERITY_HIGH,
                source_host=computer_name,
                description=(
                    f'Computer [{computer_name}] generated a process creation '
                    f'event followed by SMB activity within '
                    f'{config.RULE_5_PROCESS_TO_SMB_TIME_WINDOW} sequential '
                    f'log records. This sequence is consistent with process-'
                    f'driven lateral movement or remote service activity over SMB.'
                ),
                mitre_technique=(
                    f'{config.MITRE_COMMAND_AND_SCRIPTING} / '
                    f'{config.MITRE_SMB_LATERAL_MOVEMENT}'
                ),
                evidence_count=len(matched_ids),
                attack_label=computer_attack_label,
                matched_record_ids=matched_ids
            )

            all_generated_alerts.append(alert)
            current_alert_id += 1

    # --------------------------------------------------------
    # RULE 6: LDAP Reconnaissance
    # --------------------------------------------------------
    if not ldap_flood_triggered:
        ldap_recon_events = computer_logs[
            (computer_logs['EventID'] == config.EVENTID_NETWORK_CONNECTION) &
            (computer_logs['DestinationPortName'] == config.PORT_LDAP)
        ]

        if len(ldap_recon_events) > config.RULE_6_LDAP_RECON_THRESHOLD:
            ldap_recon_record_ids = ldap_recon_events['EventRecordID'].tolist()

            rule_6_triggered, matched_ids = check_events_in_window(
                record_id_list=ldap_recon_record_ids,
                window_size=config.RULE_6_LDAP_RECON_TIME_WINDOW,
                threshold=config.RULE_6_LDAP_RECON_THRESHOLD
            )

            if rule_6_triggered:
                alert = create_detection_alert(
                    alert_id=current_alert_id,
                    rule_number=6,
                    rule_name='LDAP Reconnaissance Activity',
                    severity=config.SEVERITY_MEDIUM,
                    source_host=computer_name,
                    description=(
                        f'Computer [{computer_name}] generated more than '
                        f'{config.RULE_6_LDAP_RECON_THRESHOLD} LDAP '
                        f'connections within '
                        f'{config.RULE_6_LDAP_RECON_TIME_WINDOW} sequential '
                        f'log records. This indicates likely LDAP-based '
                        f'reconnaissance against directory services.'
                    ),
                    mitre_technique=config.MITRE_REMOTE_SYSTEM_DISCOVERY,
                    evidence_count=len(matched_ids),
                    attack_label=computer_attack_label,
                    matched_record_ids=matched_ids
                )

                all_generated_alerts.append(alert)
                current_alert_id += 1

    # --------------------------------------------------------
    # RULE 7: RDP Credential-Based Movement
    # --------------------------------------------------------
    if not rdp_rapid_exploit_triggered:
        rdp_lateral_events = computer_logs[
            (computer_logs['EventID'] == config.EVENTID_NETWORK_CONNECTION) &
            (computer_logs['DestinationPortName'] == config.PORT_RDP)
        ]

        if len(rdp_lateral_events) > config.RULE_7_RDP_CREDENTIAL_MOVEMENT_THRESHOLD:
            rdp_lateral_record_ids = rdp_lateral_events['EventRecordID'].tolist()

            rule_7_triggered, matched_ids = check_events_in_window(
                record_id_list=rdp_lateral_record_ids,
                window_size=config.RULE_7_RDP_CREDENTIAL_MOVEMENT_TIME_WINDOW,
                threshold=config.RULE_7_RDP_CREDENTIAL_MOVEMENT_THRESHOLD
            )

            if rule_7_triggered:
                alert = create_detection_alert(
                    alert_id=current_alert_id,
                    rule_number=7,
                    rule_name='RDP Credential-Based Lateral Movement',
                    severity=config.SEVERITY_HIGH,
                    source_host=computer_name,
                    description=(
                        f'Computer [{computer_name}] generated more than '
                        f'{config.RULE_7_RDP_CREDENTIAL_MOVEMENT_THRESHOLD} '
                        f'RDP connections within '
                        f'{config.RULE_7_RDP_CREDENTIAL_MOVEMENT_TIME_WINDOW} '
                        f'sequential log records. This indicates likely '
                        f'credential-based lateral movement over RDP.'
                    ),
                    mitre_technique=config.MITRE_RDP_LATERAL_MOVEMENT,
                    evidence_count=len(matched_ids),
                    attack_label=computer_attack_label,
                    matched_record_ids=matched_ids
                )

                all_generated_alerts.append(alert)
                current_alert_id += 1

    # --------------------------------------------------------
    # RULE 8: Full Lateral Movement Kill Chain
    # --------------------------------------------------------
    kill_chain_process = computer_logs[
        computer_logs['EventID'] == config.EVENTID_PROCESS_CREATION
    ]

    kill_chain_smb = computer_logs[
        (computer_logs['EventID'] == config.EVENTID_NETWORK_CONNECTION) &
        (computer_logs['DestinationPortName'] == config.PORT_SMB)
    ]

    kill_chain_ldap = computer_logs[
        (computer_logs['EventID'] == config.EVENTID_NETWORK_CONNECTION) &
        (computer_logs['DestinationPortName'] == config.PORT_LDAP)
    ]

    kill_chain_rdp = computer_logs[
        (computer_logs['EventID'] == config.EVENTID_NETWORK_CONNECTION) &
        (computer_logs['DestinationPortName'] == config.PORT_RDP)
    ]

    if (
        len(kill_chain_process) > 0 and
        len(kill_chain_smb) > 0 and
        len(kill_chain_ldap) > 0 and
        len(kill_chain_rdp) > 0
    ):
        rule_8_triggered = False
        matched_ids = []

        for _, process_event in kill_chain_process.iterrows():
            stage_1_record_id = process_event['EventRecordID']
            chain_window_end = stage_1_record_id + config.RULE_8_KILL_CHAIN_TIME_WINDOW

            smb_after_stage_1 = kill_chain_smb[
                (kill_chain_smb['EventRecordID'] > stage_1_record_id) &
                (kill_chain_smb['EventRecordID'] <= chain_window_end)
            ]

            if len(smb_after_stage_1) == 0:
                continue

            stage_2_record_id = smb_after_stage_1['EventRecordID'].min()

            ldap_after_stage_2 = kill_chain_ldap[
                (kill_chain_ldap['EventRecordID'] > stage_2_record_id) &
                (kill_chain_ldap['EventRecordID'] <= chain_window_end)
            ]

            if len(ldap_after_stage_2) == 0:
                continue

            stage_3_record_id = ldap_after_stage_2['EventRecordID'].min()

            rdp_after_stage_3 = kill_chain_rdp[
                (kill_chain_rdp['EventRecordID'] > stage_3_record_id) &
                (kill_chain_rdp['EventRecordID'] <= chain_window_end)
            ]

            if len(rdp_after_stage_3) > 0:
                stage_4_record_id = rdp_after_stage_3['EventRecordID'].min()
                matched_ids = [
                    stage_1_record_id,
                    stage_2_record_id,
                    stage_3_record_id,
                    stage_4_record_id
                ]
                rule_8_triggered = True
                break

        if rule_8_triggered:
            alert = create_detection_alert(
                alert_id=current_alert_id,
                rule_number=8,
                rule_name='Full Lateral Movement Kill Chain',
                severity=config.SEVERITY_CRITICAL,
                source_host=computer_name,
                description=(
                    f'Computer [{computer_name}] completed a strict ordered '
                    f'sequence within {config.RULE_8_KILL_CHAIN_TIME_WINDOW} '
                    f'sequential log records: Process Creation → SMB → LDAP '
                    f'→ RDP. This indicates a full correlated lateral '
                    f'movement kill chain.'
                ),
                mitre_technique=config.MITRE_LATERAL_MOVEMENT_TACTIC,
                evidence_count=len(matched_ids),
                attack_label=computer_attack_label,
                matched_record_ids=matched_ids
            )

            all_generated_alerts.append(alert)
            current_alert_id += 1

    # --------------------------------------------------------
    # RULE 9: DNS Reconnaissance
    # --------------------------------------------------------
    dns_query_events = computer_logs[
        computer_logs['EventID'] == config.EVENTID_DNS_QUERY
    ]

    if len(dns_query_events) > config.RULE_9_DNS_RECON_THRESHOLD:
        dns_record_ids = dns_query_events['EventRecordID'].tolist()

        rule_9_triggered, matched_ids = check_events_in_window(
            record_id_list=dns_record_ids,
            window_size=config.RULE_9_DNS_RECON_TIME_WINDOW,
            threshold=config.RULE_9_DNS_RECON_THRESHOLD
        )

        if rule_9_triggered:
            alert = create_detection_alert(
                alert_id=current_alert_id,
                rule_number=9,
                rule_name='DNS Reconnaissance Activity',
                severity=config.SEVERITY_MEDIUM,
                source_host=computer_name,
                description=(
                    f'Computer [{computer_name}] generated more than '
                    f'{config.RULE_9_DNS_RECON_THRESHOLD} DNS queries '
                    f'within {config.RULE_9_DNS_RECON_TIME_WINDOW} '
                    f'sequential log records. This indicates likely '
                    f'DNS-based reconnaissance activity.'
                ),
                mitre_technique=config.MITRE_REMOTE_SYSTEM_DISCOVERY,
                evidence_count=len(matched_ids),
                attack_label=computer_attack_label,
                matched_record_ids=matched_ids
            )

            all_generated_alerts.append(alert)
            current_alert_id += 1


# ============================================================
# Step 6: Display Results
# ============================================================

print("=" * 60)
print("  STEP 6: DETECTION RESULTS                           ")
print("=" * 60)

if len(all_generated_alerts) == 0:
    print("\nNo alerts generated.")
    print("No suspicious patterns matched any of the 9 rules.")
else:
    print(f"\nTotal alerts generated : {len(all_generated_alerts)}\n")

    for alert in all_generated_alerts:
        print("=" * 60)
        print(f"  ALERT #{alert['Alert ID']}")
        print("=" * 60)
        print(f"  Rule Number        : {alert['Rule Number']}")
        print(f"  Rule Name          : {alert['Rule Name']}")
        print(f"  Severity           : {alert['Severity']}")
        print(f"  Status             : {alert['Status']}")
        print(f"  Source Host        : {alert['Source Host']}")
        print(f"  MITRE ATT&CK       : {alert['MITRE Technique']}")
        print(f"  Evidence Count     : {alert['Evidence Count']} logs")
        print(f"  Real Attack        : {alert['Is Real Attack']}")
        print(f"  Matched Record IDs : {alert['Matched Record IDs']}")
        print(f"  Detected At        : {alert['Detection Time']}")
        print(f"  Description        :")
        print(f"  {alert['Description']}")
        print()


# ============================================================
# Step 7: Statistics
# ============================================================

print("=" * 60)
print("  STEP 7: ALERT STATISTICS                            ")
print("=" * 60)

if len(all_generated_alerts) > 0:
    alerts_dataframe = pd.DataFrame(all_generated_alerts)

    print("\nAlerts by Severity Level:")
    severity_counts = alerts_dataframe['Severity'].value_counts()
    for severity_level, count in severity_counts.items():
        print(f"  {severity_level:<10} : {count:>4} alerts")

    print("\nAlerts by Rule:")
    rule_counts = alerts_dataframe['Rule Name'].value_counts()
    for rule_name, count in rule_counts.items():
        print(f"  Rule: {rule_name:<45} : {count:>4} alerts")

    true_positive_count = len(
        alerts_dataframe[alerts_dataframe['Attack Label'] == 1]
    )
    false_positive_count = len(
        alerts_dataframe[alerts_dataframe['Attack Label'] == 0]
    )
    total_alert_count = len(alerts_dataframe)

    true_positive_rate = round(
        true_positive_count / total_alert_count * 100, 2
    )
    false_positive_rate = round(
        false_positive_count / total_alert_count * 100, 2
    )

    print(f"\nAlert Accuracy Analysis:")
    print(f"  True Positives  : {true_positive_count:>4} alerts on real attacks")
    print(f"  False Positives : {false_positive_count:>4} alerts on normal traffic")
    print(f"  True Positive Rate  : {true_positive_rate}%")
    print(f"  False Positive Rate : {false_positive_rate}%")


# ============================================================
# Step 8: Save Alerts
# ============================================================

print("\n" + "=" * 60)
print("  STEP 8: SAVING ALERTS TO FILE                       ")
print("=" * 60)

if len(all_generated_alerts) > 0:
    alerts_dataframe = pd.DataFrame(all_generated_alerts)

    alerts_dataframe.to_csv(
        config.ALERTS_OUTPUT_FILE,
        index=False
    )

    print(f"\nAlerts saved to  : {config.ALERTS_OUTPUT_FILE}")
    print(f"Total alerts     : {len(alerts_dataframe)}")
    print(f"\nColumns in output file:")
    for column in alerts_dataframe.columns:
        print(f"  → {column}")
else:
    print("No alerts to save.")


# ============================================================
# Rule Engine Complete
# ============================================================

print("\n" + "=" * 60)
print("  RULE ENGINE COMPLETE                                 ")
print("=" * 60)
print(f"  Rules applied      : 9")
print(f"  Computers analyzed : {total_computers}")
print(f"  Alerts generated   : {len(all_generated_alerts)}")
print(f"  Output file        : {config.ALERTS_OUTPUT_FILE}")
print(f"  Next step          : Connect to Flask Dashboard")
print("=" * 60)