# ============================================================
# Lateral Movement Detection System
# Step 2: Feature Selection
#
# Purpose: Keep only the 8 most important features
# from the dataset as identified by the paper using
# Principal Component Analysis (PCA)
#
# Reference: Smiliotopoulos et al. (2025)
# "Assessing the detection of lateral movement through
#  unsupervised learning techniques"
# Computers & Security, Volume 149
# Table 3: The 8 selected features
# ============================================================


# ============================================================
# Import Required Libraries
# ============================================================

import pandas as pd


# ============================================================
# Step 1: Load the Dataset Again
# We reload the same balanced sample we created in EDA
# 46,000 normal rows + 4,000 attack rows = 50,000 total
# ============================================================

print("================================================")
print("            LOADING DATASET                    ")
print("================================================")
print("Please wait...")

# We will collect rows here
collected_normal_rows = []
collected_attack_rows = []

# Target number of rows
target_normal_rows = 46000
target_attack_rows = 4000

# Read file in chunks of 10,000 rows at a time
chunk_size = 10000

for chunk in pd.read_csv(
    'LMD-2023 [1.75M Elements][Labelled]checked.csv',
    chunksize=chunk_size,
    low_memory=False
):
    # Separate normal and attack rows in this chunk
    normal_rows_in_chunk = chunk[chunk['Label'] == 0]
    attack_rows_in_chunk = chunk[chunk['Label'] == 1]

    # Collect normal rows until we have enough
    if len(collected_normal_rows) < target_normal_rows:
        collected_normal_rows.append(normal_rows_in_chunk)

    # Collect attack rows until we have enough
    if len(collected_attack_rows) < target_attack_rows:
        collected_attack_rows.append(attack_rows_in_chunk)

    # Count how many we have collected so far
    current_normal_count = sum(
        len(collected_chunk)
        for collected_chunk in collected_normal_rows
    )
    current_attack_count = sum(
        len(collected_chunk)
        for collected_chunk in collected_attack_rows
    )

    # Stop reading once we have enough of both
    if (current_normal_count >= target_normal_rows and
            current_attack_count >= target_attack_rows):
        break

# Combine all collected chunks into one table
all_normal_rows = pd.concat(
    collected_normal_rows,
    ignore_index=True
)
all_attack_rows = pd.concat(
    collected_attack_rows,
    ignore_index=True
)

# Take exactly the number we need
normal_sample = all_normal_rows.head(target_normal_rows)
attack_sample = all_attack_rows.head(target_attack_rows)

# Combine normal and attack into one dataset
full_development_dataset = pd.concat(
    [normal_sample, attack_sample],
    ignore_index=True
)

# Shuffle the rows
full_development_dataset = full_development_dataset.sample(
    frac=1,
    random_state=10
).reset_index(drop=True)

print("Dataset loaded successfully!")
print(f"Total rows    : {len(full_development_dataset)}")
print(f"Total columns : {full_development_dataset.shape[1]}")


# ============================================================
# Step 2: Show Dataset BEFORE Feature Selection
# So we can compare before and after
# ============================================================

print("\n================================================")
print("       DATASET BEFORE FEATURE SELECTION        ")
print("================================================")

before_number_of_columns = full_development_dataset.shape[1]

print(f"Number of columns before : {before_number_of_columns}")
print(f"Number of rows           : {len(full_development_dataset)}")


# ============================================================
# Step 3: Define the 8 Important Features
# These are taken directly from Table 3 of the paper
# We also keep the Label column for evaluation purposes
# ============================================================

print("\n================================================")
print("    THE 8 FEATURES SELECTED FROM THE PAPER     ")
print("         Reference: Table 3                    ")
print("================================================")

# The 8 features identified by PCA in the paper
eight_selected_features = [
    'Computer',            # Which machine generated the log
    'DestinationPortName', # Which port was connected to
    'EventID',             # What type of Sysmon event occurred
    'EventRecordID',       # Sequential record number
    'Execution_ProcessID', # ID of the process that caused event
    'Initiated',           # Was connection outbound from machine
    'ProcessID',           # ID of the active process
    'SourceIsIpv6'         # Was source IP address IPv6 format
]

# Print each feature with its description
feature_descriptions = {
    'Computer':            'Which machine generated the log',
    'DestinationPortName': 'Which port was connected to (SMB/RDP/LDAP)',
    'EventID':             'What type of Sysmon event occurred',
    'EventRecordID':       'Sequential record number (tracks order)',
    'Execution_ProcessID': 'ID of the process that caused the event',
    'Initiated':           'Was the connection outbound from this machine',
    'ProcessID':           'ID of the active process at event time',
    'SourceIsIpv6':        'Was the source IP address in IPv6 format'
}

for feature_number, feature_name in enumerate(
    eight_selected_features, start=1
):
    print(
        f"Feature {feature_number}: "
        f"{feature_name:25s} → "
        f"{feature_descriptions[feature_name]}"
    )


# ============================================================
# Step 4: Define the 10 Features the Paper Dropped
# Reference: Section 2.3 of the paper
# These features were dropped because they carry
# no useful information for lateral movement detection
# ============================================================

print("\n================================================")
print("   THE 10 FEATURES DROPPED AS PER THE PAPER   ")
print("         Reference: Section 2.3               ")
print("================================================")

ten_features_to_drop = [
    'Name',          # Repeats same value across all samples
    'Guid',          # Repeats same value across all samples
    'Opcode',        # Repeats same value across all samples
    'Keywords',      # Repeats same value across all samples
    'Correlation',   # Repeats same value across all samples
    'Channel',       # Repeats same value across all samples
    'State',         # Repeats same value across all samples
    'Version',       # Repeats same value across all samples
    'StartFunction', # Repeats same value across all samples
    'ID'             # Repeats same value across all samples
]

for feature_number, feature_name in enumerate(
    ten_features_to_drop, start=1
):
    print(
        f"Dropped Feature {feature_number:02d}: "
        f"{feature_name} "
        f"→ No useful information"
    )


# ============================================================
# Step 5: Apply Feature Selection
# Keep only the 8 selected features + Label column
# Drop all other 85 columns
# ============================================================

print("\n================================================")
print("         APPLYING FEATURE SELECTION            ")
print("================================================")

# Columns to keep: 8 features + Label
columns_to_keep = eight_selected_features + ['Label']

# Apply selection - keep only these 9 columns
dataset_after_feature_selection = full_development_dataset[
    columns_to_keep
].copy()

print("Feature selection applied successfully!")


# ============================================================
# Step 6: Show Dataset AFTER Feature Selection
# Compare with before to confirm reduction
# ============================================================

print("\n================================================")
print("       DATASET AFTER FEATURE SELECTION         ")
print("================================================")

after_number_of_columns = dataset_after_feature_selection.shape[1]
number_of_columns_dropped = (
    before_number_of_columns - after_number_of_columns
)

print(f"Number of columns before : {before_number_of_columns}")
print(f"Number of columns after  : {after_number_of_columns}")
print(f"Number of columns dropped: {number_of_columns_dropped}")
print(f"Number of rows           : {len(dataset_after_feature_selection)}")

print("\nRemaining columns:")
for index, column_name in enumerate(
    dataset_after_feature_selection.columns, start=1
):
    print(f"  Column {index:02d}: {column_name}")


# ============================================================
# Step 7: Show Sample of Data After Feature Selection
# So we can see what the data looks like now
# ============================================================

print("\n================================================")
print("    SAMPLE OF DATA AFTER FEATURE SELECTION     ")
print("         (First 5 rows)                        ")
print("================================================")
print(dataset_after_feature_selection.head(5).to_string())


# ============================================================
# Step 8: Verify Label Distribution is Still Correct
# Make sure we did not lose any rows during selection
# ============================================================

print("\n================================================")
print("      LABEL DISTRIBUTION AFTER SELECTION       ")
print("================================================")

label_counts_after  = dataset_after_feature_selection[
    'Label'
].value_counts()

normal_count_after  = label_counts_after[0]
attack_count_after  = label_counts_after[1]
total_count_after   = len(dataset_after_feature_selection)

normal_percentage_after = round(
    normal_count_after / total_count_after * 100, 2
)
attack_percentage_after = round(
    attack_count_after / total_count_after * 100, 2
)

print(f"Normal rows (Label = 0) : {normal_count_after} ({normal_percentage_after}%)")
print(f"Attack rows (Label = 1) : {attack_count_after} ({attack_percentage_after}%)")
print(f"Total rows              : {total_count_after}")


# ============================================================
# Step 9: Save the Dataset After Feature Selection
# We save it so we can use it in the next step
# without having to reload and reprocess everything
# ============================================================

print("\n================================================")
print("         SAVING SELECTED FEATURES DATASET      ")
print("================================================")

output_file_name = 'dataset_after_feature_selection.csv'

dataset_after_feature_selection.to_csv(
    output_file_name,
    index=False
)

print(f"Dataset saved successfully as:")
print(f"'{output_file_name}'")
print(f"Location: your LM_Project folder")


# ============================================================
# Feature Selection Complete
# ============================================================

print("\n================================================")
print("          FEATURE SELECTION COMPLETE           ")
print("================================================")
print("Columns reduced from 94 to 9 (8 features + Label)")
print("Next step: Preprocessing (OHE + MinMax Scaling)")
print("================================================")