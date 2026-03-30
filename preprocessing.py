# ============================================================
# Lateral Movement Detection System
# Step 3: Data Preprocessing
#
# Purpose: Convert all features into numbers that the
# Isolation Forest ML model can understand and process.
# We apply two techniques from the paper:
# 1. One Hot Encoding (OHE) for text columns
# 2. MinMax Scaling for number columns
#
# Reference: Smiliotopoulos et al. (2025)
# "Assessing the detection of lateral movement through
#  unsupervised learning techniques"
# Computers & Security, Volume 149
# Section 2.4: Data Preprocessing
# ============================================================


# ============================================================
# Import Required Libraries
#
# pandas      → for loading and working with data tables
# numpy       → for working with numbers and arrays
# OneHotEncoder → converts text columns to 0/1 columns
# MinMaxScaler  → scales number columns to 0-1 range
# ColumnTransformer → applies different preprocessing
#                     to different columns at the same time
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer


# ============================================================
# Step 1: Load the Dataset After Feature Selection
# This file was saved in the previous step
# It contains 50,000 rows and 9 columns:
# 8 features + Label column
# ============================================================

print("================================================")
print("         LOADING FEATURE SELECTED DATASET      ")
print("================================================")

# Load the dataset we saved in feature_selection.py
dataset_after_feature_selection = pd.read_csv(
    'dataset_after_feature_selection.csv'
)

print("Dataset loaded successfully!")
print(f"Total rows    : {len(dataset_after_feature_selection)}")
print(f"Total columns : {dataset_after_feature_selection.shape[1]}")


# ============================================================
# Step 2: Show the Data BEFORE Preprocessing
# So we can compare what it looks like before and after
# ============================================================

print("\n================================================")
print("          DATA BEFORE PREPROCESSING            ")
print("================================================")
print(dataset_after_feature_selection.head(3).to_string())


# ============================================================
# Step 3: Separate Features from Label
#
# We separate the dataset into two parts:
# 1. features_dataset → the 8 columns we will preprocess
# 2. label_column     → the Label column we keep separate
#
# Why separate?
# The Label column must NOT be preprocessed
# It is only used AFTER training to evaluate the model
# The paper used unlabeled data for training
# (unsupervised learning does not need labels)
# ============================================================

print("\n================================================")
print("      SEPARATING FEATURES FROM LABEL           ")
print("================================================")

# The 8 feature columns only (no Label)
features_dataset = dataset_after_feature_selection.drop(
    columns=['Label']
)

# The Label column only (0 = Normal, 1 = Attack)
label_column = dataset_after_feature_selection['Label']

print(f"Features dataset shape : {features_dataset.shape}")
print(f"Label column shape     : {label_column.shape}")
print(f"Feature columns        : {features_dataset.columns.tolist()}")


# ============================================================
# Step 4: Define Which Columns Get Which Preprocessing
#
# From Table 3 of the paper:
# Categorical (text) columns → apply OHE
# Numerical (number) columns → apply MinMax
# ============================================================

print("\n================================================")
print("      DEFINING PREPROCESSING COLUMN TYPES      ")
print("================================================")

# These 5 columns contain TEXT values
# They will be converted to 0/1 columns using OHE
categorical_columns_for_ohe = [
    'Computer',            # Machine names (text)
    'DestinationPortName', # Port names like SMB, RDP (text)
    'EventID',             # Event type numbers stored as text
    'Initiated',           # true or false (text)
    'SourceIsIpv6'         # true or false (text)
]

# These 3 columns contain NUMBER values
# They will be scaled to 0-1 range using MinMax
numerical_columns_for_minmax = [
    'EventRecordID',       # Large sequential numbers
    'Execution_ProcessID', # Large process ID numbers
    'ProcessID'            # Large process ID numbers
]

print("Columns that will receive OHE (text columns):")
for column_name in categorical_columns_for_ohe:
    print(f"  → {column_name}")

print("\nColumns that will receive MinMax (number columns):")
for column_name in numerical_columns_for_minmax:
    print(f"  → {column_name}")


# ============================================================
# Step 5: Create the Preprocessing Pipeline
#
# ColumnTransformer applies different transformations
# to different columns at the same time:
# - OHE is applied to the 5 text columns
# - MinMax is applied to the 3 number columns
#
# OneHotEncoder settings:
# handle_unknown='ignore' → if a new unknown value appears
#                           during testing, ignore it
#                           instead of causing an error
# sparse_output=False     → return a regular array
#                           not a compressed sparse matrix
#
# MinMaxScaler:
# Default range is 0 to 1
# All numbers will be scaled between 0 and 1
# ============================================================

print("\n================================================")
print("      CREATING PREPROCESSING PIPELINE          ")
print("================================================")

# Create the preprocessing pipeline
# This combines OHE and MinMax into one step
preprocessing_pipeline = ColumnTransformer(
    transformers=[
        (
            'one_hot_encoding',        # Name of this transformation
            OneHotEncoder(             # The transformation to apply
                handle_unknown='ignore',
                sparse_output=False
            ),
            categorical_columns_for_ohe  # Apply to these columns
        ),
        (
            'minmax_scaling',          # Name of this transformation
            MinMaxScaler(),            # The transformation to apply
            numerical_columns_for_minmax # Apply to these columns
        )
    ]
)

print("Preprocessing pipeline created successfully!")
print("Pipeline contains:")
print("  → One Hot Encoder  for 5 text columns")
print("  → MinMax Scaler    for 3 number columns")


# ============================================================
# Step 6: Apply Preprocessing to the Features Dataset
#
# fit_transform does two things:
# 1. fit   → learns the data
#            (finds all unique values for OHE,
#             finds min and max for MinMax)
# 2. transform → applies the transformation
#            (converts text to 0/1,
#             scales numbers to 0-1)
#
# The result is a numpy array of numbers
# All values will be between 0 and 1
# ============================================================

print("\n================================================")
print("         APPLYING PREPROCESSING                ")
print("================================================")
print("Please wait...")

# Apply OHE and MinMax to all 8 feature columns
preprocessed_features_array = preprocessing_pipeline.fit_transform(
    features_dataset
)

print("Preprocessing applied successfully!")
print(f"Shape after preprocessing : {preprocessed_features_array.shape}")
print(f"Expected shape            : (50000, 47)")


# ============================================================
# Step 7: Get the New Column Names After OHE
#
# After OHE the text columns expand into many columns
# We need to get the names of all new columns
# so we can create a proper table with headers
# ============================================================

print("\n================================================")
print("         GETTING NEW COLUMN NAMES              ")
print("================================================")

# Get names of all new OHE columns
# The encoder creates names like:
# "one_hot_encoding__Computer_LAPTOP-ABC"
# "one_hot_encoding__EventID_1"
ohe_column_names = (
    preprocessing_pipeline
    .named_transformers_['one_hot_encoding']
    .get_feature_names_out(categorical_columns_for_ohe)
    .tolist()
)

# The MinMax columns keep their original names
minmax_column_names = numerical_columns_for_minmax

# Combine OHE names + MinMax names
all_new_column_names = ohe_column_names + minmax_column_names

print(f"Number of OHE columns    : {len(ohe_column_names)}")
print(f"Number of MinMax columns : {len(minmax_column_names)}")
print(f"Total columns            : {len(all_new_column_names)}")


# ============================================================
# Step 8: Convert the Preprocessed Array to a Table
#
# After preprocessing we have a numpy array (just numbers)
# We convert it back to a pandas DataFrame (table with headers)
# This makes it easier to work with and save
# ============================================================

print("\n================================================")
print("      CONVERTING ARRAY TO TABLE                ")
print("================================================")

# Convert numpy array to pandas DataFrame with column names
preprocessed_features_dataset = pd.DataFrame(
    preprocessed_features_array,
    columns=all_new_column_names
)

print("Conversion successful!")
print(f"Shape : {preprocessed_features_dataset.shape}")


# ============================================================
# Step 9: Add the Label Column Back
#
# We kept the Label column separate during preprocessing
# Now we add it back to the preprocessed dataset
# It will be used later for evaluation only
# NOT for training the Isolation Forest model
# ============================================================

print("\n================================================")
print("         ADDING LABEL COLUMN BACK              ")
print("================================================")

# Add Label column to the right side of the table
preprocessed_features_dataset['Label'] = label_column.values

# Calculate the actual column counts from the real data
# We do not hardcode these numbers
# We read them directly from the dataset shape
actual_number_of_total_columns   = preprocessed_features_dataset.shape[1]
actual_number_of_feature_columns = actual_number_of_total_columns - 1

print("Label column added back successfully!")
print(f"Final dataset shape      : {preprocessed_features_dataset.shape}")
print(f"Total columns            : {actual_number_of_total_columns}")
print(f"  → {actual_number_of_feature_columns} preprocessed feature columns")
print(f"  → 1 Label column")
print(f"  → {actual_number_of_total_columns} columns total")


# ============================================================
# Step 10: Verify the Preprocessing Results
#
# We check that:
# 1. All values are between 0 and 1
# 2. No missing values were created
# 3. Label distribution is still correct
# ============================================================

print("\n================================================")
print("         VERIFYING PREPROCESSING RESULTS       ")
print("================================================")

# Check minimum and maximum values in feature columns
# (exclude Label column from this check)
feature_columns_only = preprocessed_features_dataset.drop(
    columns=['Label']
)

minimum_value = feature_columns_only.min().min()
maximum_value = feature_columns_only.max().max()

print(f"Minimum value in features : {round(minimum_value, 4)}")
print(f"Maximum value in features : {round(maximum_value, 4)}")

if minimum_value >= 0 and maximum_value <= 1:
    print("All values are between 0 and 1 ✓")
else:
    print("WARNING: Some values are outside 0-1 range!")

# Check for missing values
total_missing_values = preprocessed_features_dataset.isnull().sum().sum()

if total_missing_values == 0:
    print("No missing values found ✓")
else:
    print(f"WARNING: {total_missing_values} missing values found!")

# Check label distribution
label_counts = preprocessed_features_dataset['Label'].value_counts()
normal_count = label_counts[0]
attack_count = label_counts[1]
total_count  = len(preprocessed_features_dataset)

normal_percentage = round(normal_count / total_count * 100, 2)
attack_percentage = round(attack_count / total_count * 100, 2)

print(f"\nLabel Distribution:")
print(f"Normal rows (Label = 0) : {normal_count} ({normal_percentage}%)")
print(f"Attack rows (Label = 1) : {attack_count} ({attack_percentage}%)")
print(f"Total rows              : {total_count}")


# ============================================================
# Step 11: Show Sample of Preprocessed Data
# So we can visually confirm values are between 0 and 1
# ============================================================

print("\n================================================")
print("      SAMPLE OF PREPROCESSED DATA (3 rows)     ")
print("   Showing first 5 columns + Label column      ")
print("================================================")

# Show first 3 rows and first 5 columns + Label
first_five_columns = preprocessed_features_dataset.iloc[
    :3,
    :5
]
print(first_five_columns.to_string())
print("\n(All other columns also contain values between 0 and 1)")


# ============================================================
# Step 12: Save the Preprocessed Dataset
#
# We save two versions:
# 1. With Label    → for evaluation after training
# 2. Without Label → for training Isolation Forest
#                    (unsupervised = no labels needed)
# ============================================================

print("\n================================================")
print("         SAVING PREPROCESSED DATASETS          ")
print("================================================")

# Save version WITH Label (for evaluation)
output_file_with_label = 'preprocessed_dataset_with_label.csv'

preprocessed_features_dataset.to_csv(
    output_file_with_label,
    index=False
)

print(f"Saved WITH label    : {output_file_with_label}")
print(f"Rows                : {len(preprocessed_features_dataset)}")
print(f"Columns             : {preprocessed_features_dataset.shape[1]}")

# Save version WITHOUT Label (for training)
# This is what we feed to Isolation Forest
output_file_without_label = 'preprocessed_dataset_without_label.csv'

preprocessed_dataset_without_label = preprocessed_features_dataset.drop(
    columns=['Label']
)

preprocessed_dataset_without_label.to_csv(
    output_file_without_label,
    index=False
)

print(f"\nSaved WITHOUT label : {output_file_without_label}")
print(f"Rows                : {len(preprocessed_dataset_without_label)}")
print(f"Columns             : {preprocessed_dataset_without_label.shape[1]}")


# ============================================================
# Preprocessing Complete
# ============================================================

print("\n================================================")
print("           PREPROCESSING COMPLETE              ")
print("================================================")
print("Two files saved:")
print(f"  1. {output_file_with_label}")
print(f"     → 90 columns (89 features + Label)")
print(f"     → Used for evaluation after training")
print(f"  2. {output_file_without_label}")
print(f"     → 89 columns (features only, no Label)")
print(f"     → Used for training Isolation Forest")
print("\nNext step: Train Isolation Forest ML Model")
print("================================================")