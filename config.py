# ============================================================
# Lateral Movement Detection System
# Configuration File: config.py
#
# Purpose:
# Central configuration for the rule-based correlation engine.
# All thresholds, port names, event IDs, file paths, severity
# levels, and MITRE references are defined here.
#
# Notes:
# - EventRecordID is used as a proxy for time because the
#   dataset after feature selection does not contain a
#   timestamp column.
# - Thresholds are triggered when event count is GREATER THAN
#   the configured threshold inside the configured window.
# ============================================================


# ============================================================
# Dataset / Output Files
# ============================================================

DATASET_FILE_PATH = 'dataset_after_feature_selection.csv'
ALERTS_OUTPUT_FILE = 'rule_engine_alerts.csv'


# ============================================================
# Required Columns
# ============================================================

REQUIRED_COLUMNS = [
    'Computer',
    'EventID',
    'DestinationPortName',
    'EventRecordID',
    'Initiated',
    'ProcessID',
    'Label'
]


# ============================================================
# Sysmon Event IDs
# ============================================================

EVENTID_PROCESS_CREATION = 1
EVENTID_NETWORK_CONNECTION = 3
EVENTID_PROCESS_ACCESS = 10
EVENTID_DNS_QUERY = 22


# ============================================================
# DestinationPortName Values
# ============================================================

PORT_SMB = 'microsoft-ds'      # Port 445
PORT_RDP = 'ms-wbt-server'     # Port 3389
PORT_LDAP = 'ldap'             # Port 389
PORT_DNS = 'domain'            # Port 53


# ============================================================
# Record-Based Time Windows
# ============================================================

TIME_WINDOW_VERY_RAPID_RECORDS = 30
TIME_WINDOW_RAPID_RECORDS = 60
TIME_WINDOW_MODERATE_RECORDS = 120
TIME_WINDOW_SLOW_RECORDS = 300


# ============================================================
# Rule Thresholds / Windows
# Trigger condition = count > threshold
# ============================================================

# Rule 1: SMB rapid scan / rapid SMB exploitation behavior
RULE_1_SMB_RAPID_SCAN_THRESHOLD = 3
RULE_1_SMB_RAPID_SCAN_TIME_WINDOW = TIME_WINDOW_RAPID_RECORDS

# Rule 2: RDP rapid exploitation pattern
RULE_2_RDP_RAPID_EXPLOIT_THRESHOLD = 3
RULE_2_RDP_RAPID_EXPLOIT_TIME_WINDOW = TIME_WINDOW_MODERATE_RECORDS

# Rule 3: LDAP flood pattern
RULE_3_LDAP_FLOOD_THRESHOLD = 10
RULE_3_LDAP_FLOOD_TIME_WINDOW = TIME_WINDOW_VERY_RAPID_RECORDS

# Rule 4: Suspicious process access activity
RULE_4_PROCESS_ACCESS_THRESHOLD = 2

# Rule 5: Process creation followed by SMB activity
RULE_5_PROCESS_TO_SMB_TIME_WINDOW = TIME_WINDOW_MODERATE_RECORDS

# Rule 6: LDAP reconnaissance
RULE_6_LDAP_RECON_THRESHOLD = 5
RULE_6_LDAP_RECON_TIME_WINDOW = TIME_WINDOW_RAPID_RECORDS

# Rule 7: RDP credential-based movement
RULE_7_RDP_CREDENTIAL_MOVEMENT_THRESHOLD = 3
RULE_7_RDP_CREDENTIAL_MOVEMENT_TIME_WINDOW = TIME_WINDOW_SLOW_RECORDS

# Rule 8: Full lateral movement kill chain
RULE_8_KILL_CHAIN_TIME_WINDOW = TIME_WINDOW_SLOW_RECORDS

# Rule 9: DNS reconnaissance
RULE_9_DNS_RECON_THRESHOLD = 10
RULE_9_DNS_RECON_TIME_WINDOW = TIME_WINDOW_RAPID_RECORDS


# ============================================================
# Alert Severity Levels
# ============================================================

SEVERITY_CRITICAL = 'CRITICAL'
SEVERITY_HIGH = 'HIGH'
SEVERITY_MEDIUM = 'MEDIUM'
SEVERITY_LOW = 'LOW'


# ============================================================
# Alert Status Values
# ============================================================

STATUS_UNRESOLVED = 'Unresolved'
STATUS_IN_PROGRESS = 'In Progress'
STATUS_RESOLVED = 'Resolved'


# ============================================================
# MITRE ATT&CK References
# ============================================================

MITRE_EXPLOITATION_REMOTE_SERVICES = 'T1210 - Exploitation of Remote Services'
MITRE_OS_CREDENTIAL_DUMPING = 'T1003 - OS Credential Dumping'
MITRE_COMMAND_AND_SCRIPTING = 'T1059 - Command and Scripting Interpreter'
MITRE_SMB_LATERAL_MOVEMENT = 'T1021.002 - Remote Services: SMB/Windows Admin Shares'
MITRE_RDP_LATERAL_MOVEMENT = 'T1021.001 - Remote Services: Remote Desktop Protocol'
MITRE_REMOTE_SYSTEM_DISCOVERY = 'T1018 - Remote System Discovery'
MITRE_LATERAL_MOVEMENT_TACTIC = 'TA0008 - Lateral Movement'