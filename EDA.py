# ============================================================
# Lateral Movement Detection System
# Step 1: Exploratory Data Analysis (EDA)
#
# Purpose: Load the LMD-2023 dataset and verify its contents
# before applying any preprocessing or machine learning
#
# Reference: Smiliotopoulos et al. (2025)
# "Assessing the detection of lateral movement through
#  unsupervised learning techniques"
# Computers & Security, Volume 149
# ============================================================


# ============================================================
# Import Required Libraries
# ============================================================

import pandas as pd


# ============================================================
# Step 1: Load the Dataset in Chunks
# We read the file in small pieces to save memory
# Then we collect normal and attack rows separately
# ============================================================

print("================================================")
print("            LOADING DATASET                    ")
print("================================================")
print("Please wait...")

# We will collect rows here
collected_normal_rows = []
collected_attack_rows = []

# How many rows we want in our final sample
target_normal_rows = 46000   # 92% of 50,000
target_attack_rows = 4000    # 8%  of 50,000

# Read the file in chunks of 10,000 rows at a time
# This prevents loading the full 1GB into memory
chunk_size = 10000

for chunk in pd.read_csv(
    'LMD-2023 [1.75M Elements][Labelled]checked.csv',
    chunksize=chunk_size,
    low_memory=False
):
    # Separate normal and attack rows in this chunk
    normal_in_chunk = chunk[chunk['Label'] == 0]
    attack_in_chunk = chunk[chunk['Label'] == 1]

    # Collect normal rows until we have enough
    if len(collected_normal_rows) < target_normal_rows:
        collected_normal_rows.append(normal_in_chunk)

    # Collect attack rows until we have enough
    if len(collected_attack_rows) < target_attack_rows:
        collected_attack_rows.append(attack_in_chunk)

    # Stop reading once we have enough of both
    current_normal_count = sum(
        len(chunk) for chunk in collected_normal_rows
    )
    current_attack_count = sum(
        len(chunk) for chunk in collected_attack_rows
    )

    print(
        f"Collected so far → "
        f"Normal: {current_normal_count} / {target_normal_rows} | "
        f"Attack: {current_attack_count} / {target_attack_rows}"
    )

    if (current_normal_count >= target_normal_rows and
            current_attack_count >= target_attack_rows):
        print("Target sample size reached. Stopping.")
        break


# ============================================================
# Step 2: Combine Normal and Attack Rows
# ============================================================

print("\n================================================")
print("         COMBINING NORMAL AND ATTACK ROWS      ")
print("================================================")

# Combine all collected normal rows into one table
all_normal_rows = pd.concat(
    collected_normal_rows,
    ignore_index=True
)

# Combine all collected attack rows into one table
all_attack_rows = pd.concat(
    collected_attack_rows,
    ignore_index=True
)

# Take exactly the number we need
normal_sample = all_normal_rows.head(target_normal_rows)
attack_sample = all_attack_rows.head(target_attack_rows)

# Combine normal and attack into one dataset
development_dataset = pd.concat(
    [normal_sample, attack_sample],
    ignore_index=True
)

# Shuffle rows so normal and attack are mixed together
development_dataset = development_dataset.sample(
    frac=1,
    random_state=10
).reset_index(drop=True)

print("Dataset combined and shuffled successfully!")


# ============================================================
# Step 3: Verify the Sample Size and Label Distribution
# ============================================================

print("\n================================================")
print("         DEVELOPMENT SAMPLE INFORMATION        ")
print("================================================")

sample_label_counts  = development_dataset['Label'].value_counts()
sample_normal_count  = sample_label_counts[0]
sample_attack_count  = sample_label_counts[1]
sample_total         = len(development_dataset)

sample_normal_percentage = round(
    sample_normal_count / sample_total * 100, 2
)
sample_attack_percentage = round(
    sample_attack_count / sample_total * 100, 2
)

print(f"Normal rows (Label = 0) : {sample_normal_count} ({sample_normal_percentage}%)")
print(f"Attack rows (Label = 1) : {sample_attack_count} ({sample_attack_percentage}%)")
print(f"Total rows              : {sample_total}")
print(f"Number of columns       : {development_dataset.shape[1]}")


# ============================================================
# Step 4: Show All Column Names
# ============================================================

print("\n================================================")
print("              ALL COLUMN NAMES                 ")
print("================================================")

all_column_names = development_dataset.columns.tolist()

for index, column_name in enumerate(all_column_names):
    print(f"Column {index + 1:02d}: {column_name}")


# ============================================================
# Step 5: Check for Missing Values
# ============================================================

print("\n================================================")
print("           MISSING VALUES CHECK                ")
print("================================================")

missing_values_per_column = development_dataset.isnull().sum()
columns_that_have_missing = missing_values_per_column[
    missing_values_per_column > 0
]

if len(columns_that_have_missing) == 0:
    print("No missing values found. Dataset is clean.")
else:
    print("Columns with missing values:")
    print(columns_that_have_missing)


# ============================================================
# Step 6: Verify the 8 Important Features From the Paper
# Reference: Table 3 in Smiliotopoulos et al. (2025)
# ============================================================

print("\n================================================")
print("    VERIFYING THE 8 PAPER FEATURES EXIST       ")
print("================================================")

eight_important_features = [
    'Computer',
    'DestinationPortName',
    'EventID',
    'EventRecordID',
    'Execution_ProcessID',
    'Initiated',
    'ProcessID',
    'SourceIsIpv6'
]

all_features_found = True

for feature_name in eight_important_features:
    if feature_name in development_dataset.columns:
        print(f"FOUND   : {feature_name}")
    else:
        print(f"MISSING : {feature_name}")
        all_features_found = False

if all_features_found:
    print("\nAll 8 features from the paper are present.")
else:
    print("\nSome features are missing. Check column names above.")


# ============================================================
# Exploratory Data Analysis Complete
# ============================================================

print("\n================================================")
print("      EXPLORATORY DATA ANALYSIS COMPLETE       ")
print("================================================")
print("Next step: Feature Selection and Preprocessing")
print("================================================")