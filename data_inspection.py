import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_folder = "data"

file_path = "data/Monday-WorkingHours.pcap_ISCX.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Rows and columns before cleaning:", df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows before cleaning:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

print("\nRows and columns after cleaning:", df.shape)
print("\nLabel distribution:")
print(df[" Label"].value_counts())
print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nBasic statistical summary:")
print(df.describe())
print("\nColumns containing missing values:")

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])
print("\nRows with missing Flow Bytes/s:")
print(df[df["Flow Bytes/s"].isnull()][["Flow Bytes/s", " Label"]])
print("\nInfinite values:")
print(df.select_dtypes(include="number").isin([float("inf"), float("-inf")]).sum().sum())
print("\nColumns containing infinite values:")

infinite_counts = df.select_dtypes(include="number").isin(
    [float("inf"), float("-inf")]
).sum()

print(infinite_counts[infinite_counts > 0])
print("\nRows with infinite Flow Bytes/s or Flow Packets/s:")

infinite_rows = df[
    df["Flow Bytes/s"].isin([float("inf"), float("-inf")]) |
    df[" Flow Packets/s"].isin([float("inf"), float("-inf")])
]

print(infinite_rows.iloc[:, [1, 2, 3, 5, 14, 15, 78]].head(20))
print("\nZero Flow Duration rows:")
print((df[" Flow Duration"] == 0).sum())
print("\nZero-duration rows with problematic rate values:")

zero_duration = df[" Flow Duration"] == 0

print(
    df.loc[
        zero_duration,
        ["Flow Bytes/s", " Flow Packets/s"]
    ].isnull().sum()
)

print(
    "Infinite Flow Bytes/s:",
    df.loc[zero_duration, "Flow Bytes/s"].isin([float("inf"), float("-inf")]).sum()
)

print(
    "Infinite Flow Packets/s:",
    df.loc[zero_duration, " Flow Packets/s"].isin([float("inf"), float("-inf")]).sum()
)
print("\nLabels of zero-duration rows:")
print(df.loc[df[" Flow Duration"] == 0, " Label"].value_counts())
print("\nProblematic values before cleaning:")

print("Flow Bytes/s:")
print(df["Flow Bytes/s"].value_counts(dropna=False).head())

print("\nFlow Packets/s:")
print(df[" Flow Packets/s"].value_counts(dropna=False).head())
print("\nCleaning infinite and missing rate values...")

# Replace infinite values with NaN
df["Flow Bytes/s"] = df["Flow Bytes/s"].replace([float("inf"), float("-inf")], float("nan"))
df[" Flow Packets/s"] = df[" Flow Packets/s"].replace([float("inf"), float("-inf")], float("nan"))

# Replace NaN values with 0
df["Flow Bytes/s"] = df["Flow Bytes/s"].fillna(0)
df[" Flow Packets/s"] = df[" Flow Packets/s"].fillna(0)

print("Cleaning completed.")
print("\nAfter cleaning:")

print("Missing values in Flow Bytes/s:",
      df["Flow Bytes/s"].isnull().sum())

print("Missing values in Flow Packets/s:",
      df[" Flow Packets/s"].isnull().sum())

print("Infinite values in Flow Bytes/s:",
      df["Flow Bytes/s"].isin([float("inf"), float("-inf")]).sum())

print("Infinite values in Flow Packets/s:",
      df[" Flow Packets/s"].isin([float("inf"), float("-inf")]).sum())
print("\nRemaining missing values after cleaning:")

remaining_missing = df.isnull().sum()

print(remaining_missing[remaining_missing > 0])
print("\nNegative Flow Duration values:")
print((df[" Flow Duration"] < 0).sum())
print("\nRows with negative Flow Duration:")

negative_duration = df[" Flow Duration"] < 0

print(
    df.loc[
        negative_duration,
        [" Flow Duration", " Label"]
    ]
)
print("\nDetailed inspection of negative Flow Duration rows:")

negative_duration = df[" Flow Duration"] < 0

print(df.loc[negative_duration].iloc[:, [1, 2, 3, 5, 14, 15, 78]])
print("\nCleaning negative Flow Duration values...")

negative_duration = df[" Flow Duration"] < 0

print("Negative duration rows before cleaning:", negative_duration.sum())

# Replace negative duration with 0
df.loc[negative_duration, " Flow Duration"] = 0

# Negative rate values associated with these invalid durations
df.loc[df["Flow Bytes/s"] < 0, "Flow Bytes/s"] = 0
df.loc[df[" Flow Packets/s"] < 0, " Flow Packets/s"] = 0

print("Negative duration rows after cleaning:",
      (df[" Flow Duration"] < 0).sum())

print("Negative Flow Bytes/s after cleaning:",
      (df["Flow Bytes/s"] < 0).sum())

print("Negative Flow Packets/s after cleaning:",
      (df[" Flow Packets/s"] < 0).sum())
print("\n=== FINAL CLEANING VERIFICATION ===")

print("Total missing values:",
      df.isnull().sum().sum())

print("Total infinite values:",
      df.select_dtypes(include="number").isin(
          [float("inf"), float("-inf")]
      ).sum().sum())

print("Negative Flow Duration:",
      (df[" Flow Duration"] < 0).sum())

print("Negative Flow Bytes/s:",
      (df["Flow Bytes/s"] < 0).sum())

print("Negative Flow Packets/s:",
      (df[" Flow Packets/s"] < 0).sum())

print("\nFinal dataset shape:", df.shape)

print("\nFinal label distribution:")
print(df[" Label"].value_counts())
print("\n=== LABEL DISTRIBUTION OF ALL CICIDS2017 RAW FILES ===")

for filename in sorted(os.listdir(data_folder)):

    # Skip the cleaned combined dataset
    if filename == "CICIDS2017_cleaned.csv":
        continue

    if filename.endswith(".csv"):

        file_path = os.path.join(data_folder, filename)

        temp_df = pd.read_csv(file_path)

        print("\n" + "=" * 80)
        print(filename)
        print("Rows:", len(temp_df))

        print("\nLabels:")
        print(temp_df[" Label"].value_counts())
print("\n=== COMPLETE CICIDS2017 LABEL SUMMARY ===")

all_label_counts = {}

for filename in sorted(os.listdir(data_folder)):

    # Skip the cleaned combined dataset
    if filename == "CICIDS2017_cleaned.csv":
        continue

    if filename.endswith(".csv"):

        file_path = os.path.join(data_folder, filename)

        temp_df = pd.read_csv(file_path)

        label_counts = temp_df[" Label"].value_counts()

        for label, count in label_counts.items():
            all_label_counts[label] = all_label_counts.get(label, 0) + count

print("\nCombined label counts:")
for label, count in sorted(
    all_label_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{label}: {count}")
print("\nTotal rows by label across all files:")

for label, count in sorted(
    all_label_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{label}: {count}")

print("\n=== OVERALL CLASS DISTRIBUTION ===")

total_rows = sum(all_label_counts.values())

print("Total rows across all files:", total_rows)

for label, count in sorted(
    all_label_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    percentage = (count / total_rows) * 100
    print(f"{label}: {count} ({percentage:.4f}%)")

print("\n=== COMBINING ALL CICIDS2017 FILES ===")

all_data = []

for filename in sorted(os.listdir(data_folder)):
    if filename == "CICIDS2017_cleaned.csv":
        continue

    if filename.endswith(".csv"):
        file_path = os.path.join(data_folder, filename)

        print("Loading:", filename)

        temp_df = pd.read_csv(file_path)
        all_data.append(temp_df)

df_all = pd.concat(all_data, ignore_index=True)

print("\nAll files combined successfully!")
print("Combined dataset shape:", df_all.shape)
print("\n=== COMBINED LABEL DISTRIBUTION ===")

print(df_all[" Label"].value_counts())
print("\n=== WEB ATTACK LABEL CHECK ===")

for label in df_all[" Label"].dropna().unique():

    if "Web Attack" in str(label):
        print("Web attack label:", label)

print("\n=== CLEANING WEB ATTACK LABELS ===")

df_all[" Label"] = df_all[" Label"].replace({
    "Web Attack � Brute Force": "Web Attack - Brute Force",
    "Web Attack � XSS": "Web Attack - XSS",
    "Web Attack � Sql Injection": "Web Attack - Sql Injection"
})

print("Web Attack labels cleaned.")

print("\nUpdated labels:")
print(df_all[" Label"].value_counts())

print("\n=== COMBINED DATA QUALITY CHECK ===")

print("Missing values:",
      df_all.isnull().sum().sum())

print("Infinite values:",
      df_all.select_dtypes(include="number").isin(
          [float("inf"), float("-inf")]
      ).sum().sum())
print("\n=== COMBINED DATA PROBLEM COLUMNS ===")

# Missing values by column
missing_all = df_all.isnull().sum()

print("\nColumns with missing values:")
print(missing_all[missing_all > 0])

# Infinite values by column
infinite_all = df_all.select_dtypes(include="number").isin(
    [float("inf"), float("-inf")]
).sum()

print("\nColumns with infinite values:")
print(infinite_all[infinite_all > 0])
print("\n=== CLEANING COMBINED DATASET ===")

# Replace infinite values with NaN
df_all["Flow Bytes/s"] = df_all["Flow Bytes/s"].replace(
    [float("inf"), float("-inf")],
    float("nan")
)

df_all[" Flow Packets/s"] = df_all[" Flow Packets/s"].replace(
    [float("inf"), float("-inf")],
    float("nan")
)

# Replace missing rate values with 0
df_all["Flow Bytes/s"] = df_all["Flow Bytes/s"].fillna(0)

df_all[" Flow Packets/s"] = df_all[" Flow Packets/s"].fillna(0)

print("Rate columns cleaned.")
print("\n=== VERIFYING COMBINED DATA CLEANING ===")

print(
    "Missing Flow Bytes/s:",
    df_all["Flow Bytes/s"].isnull().sum()
)

print(
    "Missing Flow Packets/s:",
    df_all[" Flow Packets/s"].isnull().sum()
)

print(
    "Infinite Flow Bytes/s:",
    df_all["Flow Bytes/s"].isin(
        [float("inf"), float("-inf")]
    ).sum()
)

print(
    "Infinite Flow Packets/s:",
    df_all[" Flow Packets/s"].isin(
        [float("inf"), float("-inf")]
    ).sum()
)
print("\n=== COMBINED DATASET DUPLICATE CHECK ===")

print("Duplicate rows before cleaning:",
      df_all.duplicated().sum())
print("\n=== DUPLICATE ROW LABEL DISTRIBUTION ===")

duplicate_rows = df_all[df_all.duplicated(keep=False)]

print(
    duplicate_rows[" Label"].value_counts()
)
print("\n=== UNIQUE ROW COUNT ===")

print(
    "Rows after removing exact duplicates:",
    df_all.drop_duplicates().shape[0]
)

print(
    "Rows that would be removed:",
    df_all.shape[0] - df_all.drop_duplicates().shape[0]
)
print("\n=== REMOVING DUPLICATES FROM COMBINED DATASET ===")

before_duplicates = len(df_all)

df_all = df_all.drop_duplicates().reset_index(drop=True)

after_duplicates = len(df_all)

print("Rows before removing duplicates:", before_duplicates)
print("Rows after removing duplicates:", after_duplicates)
print("Rows removed:", before_duplicates - after_duplicates)

print("\nRemaining duplicate rows:",
      df_all.duplicated().sum())
print("\n=== LABEL DISTRIBUTION AFTER DUPLICATE REMOVAL ===")

label_counts_after = df_all[" Label"].value_counts()

print(label_counts_after)

print("\n=== CLASS PERCENTAGES AFTER DUPLICATE REMOVAL ===")

total_rows_after = len(df_all)

for label, count in label_counts_after.items():
    percentage = (count / total_rows_after) * 100
    print(f"{label}: {count} ({percentage:.4f}%)")

print("\n=== CHECKING NEGATIVE FLOW DURATION ===")

negative_duration_all = df_all[" Flow Duration"] < 0

print(
    "Negative Flow Duration rows:",
    negative_duration_all.sum()
)

if negative_duration_all.sum() > 0:
    print("\nLabels of negative-duration rows:")
    print(
        df_all.loc[
            negative_duration_all,
            " Label"
        ].value_counts()
    )

print("\n=== CLEANING NEGATIVE FLOW DURATION ===")

negative_duration_all = df_all[" Flow Duration"] < 0

print(
    "Negative duration rows before cleaning:",
    negative_duration_all.sum()
)

# Replace negative duration with 0
df_all.loc[
    negative_duration_all,
    " Flow Duration"
] = 0

# Replace negative rate values with 0
df_all.loc[
    df_all["Flow Bytes/s"] < 0,
    "Flow Bytes/s"
] = 0

df_all.loc[
    df_all[" Flow Packets/s"] < 0,
    " Flow Packets/s"
] = 0

print(
    "Negative duration rows after cleaning:",
    (df_all[" Flow Duration"] < 0).sum()
)

print(
    "Negative Flow Bytes/s after cleaning:",
    (df_all["Flow Bytes/s"] < 0).sum()
)

print(
    "Negative Flow Packets/s after cleaning:",
    (df_all[" Flow Packets/s"] < 0).sum()
)
print("\n=== STANDARDIZING COLUMN NAMES ===")

# Remove leading/trailing spaces from all column names
df_all.columns = df_all.columns.str.strip()

print("Column names standardized.")

print("\nFirst 10 column names:")
print(df_all.columns[:10].tolist())

print("\nLabel column:")
print("Label" in df_all.columns)
print("\n=== CREATING STABLE ROW IDs ===")

df_all.insert(0, "Row_ID", range(len(df_all)))

print("Row IDs created.")

print("\nFirst 5 Row IDs:")
print(df_all["Row_ID"].head().tolist())

print("\nLast 5 Row IDs:")
print(df_all["Row_ID"].tail().tolist())

print("\nTotal Row IDs:")
print(df_all["Row_ID"].nunique())

print("\nTotal rows:")
print(len(df_all))
print("\n=== SAVING CLEANED DATASET ===")

output_file = "data/CICIDS2017_cleaned.csv"

df_all.to_csv(output_file, index=False)

print("Cleaned dataset saved successfully!")
print("File:", output_file)
print("Rows:", len(df_all))
print("Columns:", len(df_all.columns))
print("\n=== SEPARATING FEATURES AND TARGET ===")

# Keep Row_ID separately for tracking
row_ids = df_all["Row_ID"]

# Target variable
y = df_all["Label"]

# Feature variables
X = df_all.drop(columns=["Label", "Row_ID"])

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nNumber of features:", X.shape[1])
print("Number of target values:", y.shape[0])

print("\nTarget classes:")
print(y.value_counts())
print("\n=== FINAL SAVED DATASET VALIDATION ===")

# Load the saved cleaned dataset
df_check = pd.read_csv("data/CICIDS2017_cleaned.csv")

print("Saved dataset shape:", df_check.shape)

print("Total rows:", len(df_check))
print("Total columns:", len(df_check.columns))

print("\nMissing values:")
print(df_check.isnull().sum().sum())

print("\nDuplicate rows:")
print(df_check.duplicated().sum())

print("\nRow_ID uniqueness:")
print(df_check["Row_ID"].nunique())

print("\nLabel column exists:")
print("Label" in df_check.columns)

print("\nNumber of classes:")
print(df_check["Label"].nunique())

print("\nFinal class distribution:")
print(df_check["Label"].value_counts())
print("\n=== FINAL NUMERIC VALIDATION ===")

numeric_columns = df_check.select_dtypes(include="number").columns

print(
    "Total infinite values:",
    df_check[numeric_columns].isin(
        [float("inf"), float("-inf")]
    ).sum().sum()
)

print(
    "Negative Flow Duration:",
    (df_check["Flow Duration"] < 0).sum()
)

print(
    "Negative Flow Bytes/s:",
    (df_check["Flow Bytes/s"] < 0).sum()
)

print(
    "Negative Flow Packets/s:",
    (df_check["Flow Packets/s"] < 0).sum()
)
print("\n=== CREATING TRAIN / VALIDATION / TEST SPLITS ===")

# Separate features and target
X = df_check.drop(columns=["Label", "Row_ID"])
y = df_check["Label"]
row_ids = df_check["Row_ID"]

# First split: 80% train+validation, 20% test
X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
    X,
    y,
    row_ids,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split: 75% of the remaining 80% = 60% total train
X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
    X_temp,
    y_temp,
    ids_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("\nSplit sizes:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

print("\nExpected proportions:")
print("Training: 60%")
print("Validation: 20%")
print("Test: 20%")

print("\nActual proportions:")
print(f"Training:   {len(X_train) / len(df_check):.2%}")
print(f"Validation: {len(X_val) / len(df_check):.2%}")
print(f"Test:       {len(X_test) / len(df_check):.2%}")

print("\nClass distribution in training set:")
print(y_train.value_counts())

print("\nClass distribution in validation set:")
print(y_val.value_counts())

print("\nClass distribution in test set:")
print(y_test.value_counts())
print("\n=== VERIFYING SPLIT OVERLAP ===")

train_ids_set = set(ids_train)
val_ids_set = set(ids_val)
test_ids_set = set(ids_test)

train_val_overlap = train_ids_set.intersection(val_ids_set)
train_test_overlap = train_ids_set.intersection(test_ids_set)
val_test_overlap = val_ids_set.intersection(test_ids_set)

print("Train ∩ Validation:", len(train_val_overlap))
print("Train ∩ Test:", len(train_test_overlap))
print("Validation ∩ Test:", len(val_test_overlap))

if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):
    print("PASS: No Row_ID overlap between train, validation, and test sets.")
else:
    print("FAIL: Overlap detected between splits.")
    print("\n=== SAVING TRAIN / VALIDATION / TEST SPLITS ===")

# Create splits folder
os.makedirs("data/splits", exist_ok=True)

# Save training split
train_data = X_train.copy()
train_data["Label"] = y_train.values
train_data["Row_ID"] = ids_train.values
train_data.to_csv("data/splits/train.csv", index=False)

# Save validation split
val_data = X_val.copy()
val_data["Label"] = y_val.values
val_data["Row_ID"] = ids_val.values
val_data.to_csv("data/splits/validation.csv", index=False)

# Save test split
test_data = X_test.copy()
test_data["Label"] = y_test.values
test_data["Row_ID"] = ids_test.values
test_data.to_csv("data/splits/test.csv", index=False)

print("Training split saved successfully.")
print("Validation split saved successfully.")
print("Test split saved successfully.")

print("\nSaved files:")
print("data/splits/train.csv")
print("data/splits/validation.csv")
print("data/splits/test.csv")
print("PASS: No Row_ID overlap between train, validation, and test sets.")
print("\n=== SAVING TRAIN / VALIDATION / TEST SPLITS ===")

# Create splits folder
os.makedirs("data/splits", exist_ok=True)

# Save training split
train_data = X_train.copy()
train_data["Label"] = y_train.values
train_data["Row_ID"] = ids_train.values
train_data.to_csv("data/splits/train.csv", index=False)

# Save validation split
val_data = X_val.copy()
val_data["Label"] = y_val.values
val_data["Row_ID"] = ids_val.values
val_data.to_csv("data/splits/validation.csv", index=False)

# Save test split
test_data = X_test.copy()
test_data["Label"] = y_test.values
test_data["Row_ID"] = ids_test.values
test_data.to_csv("data/splits/test.csv", index=False)

print("Training split saved successfully.")
print("Validation split saved successfully.")
print("Test split saved successfully.")

print("\nSaved files:")
print("data/splits/train.csv")
print("data/splits/validation.csv")
print("data/splits/test.csv")
print("\n=== VALIDATING SAVED SPLIT FILES ===")

# Reload saved split files
train_check = pd.read_csv("data/splits/train.csv")
val_check = pd.read_csv("data/splits/validation.csv")
test_check = pd.read_csv("data/splits/test.csv")

print("\nSaved split shapes:")
print("Training:", train_check.shape)
print("Validation:", val_check.shape)
print("Test:", test_check.shape)

# Check required columns
print("\nLabel column present:")
print("Training:", "Label" in train_check.columns)
print("Validation:", "Label" in val_check.columns)
print("Test:", "Label" in test_check.columns)

print("\nRow_ID column present:")
print("Training:", "Row_ID" in train_check.columns)
print("Validation:", "Row_ID" in val_check.columns)
print("Test:", "Row_ID" in test_check.columns)

# Check Row_ID uniqueness
print("\nRow_ID uniqueness:")
print("Training:", train_check["Row_ID"].nunique() == len(train_check))
print("Validation:", val_check["Row_ID"].nunique() == len(val_check))
print("Test:", test_check["Row_ID"].nunique() == len(test_check))

# Check overlap after reloading
train_ids = set(train_check["Row_ID"])
val_ids = set(val_check["Row_ID"])
test_ids = set(test_check["Row_ID"])

print("\nOverlap after reloading:")
print("Train ∩ Validation:", len(train_ids.intersection(val_ids)))
print("Train ∩ Test:", len(train_ids.intersection(test_ids)))
print("Validation ∩ Test:", len(val_ids.intersection(test_ids)))

# Check total rows
total_split_rows = len(train_check) + len(val_check) + len(test_check)

print("\nTotal rows across saved splits:", total_split_rows)
print("Original cleaned dataset rows:", len(df_check))

if (
    len(train_check) == 1513416
    and len(val_check) == 504473
    and len(test_check) == 504473
    and total_split_rows == len(df_check)
    and len(train_ids.intersection(val_ids)) == 0
    and len(train_ids.intersection(test_ids)) == 0
    and len(val_ids.intersection(test_ids)) == 0
):
    print("\nPASS: Saved split files validated successfully.")
else:
    print("\nFAIL: Problem detected in saved split files.")
    print("\n=== PREPARING FOR LEAKAGE-SAFE PREPROCESSING ===")

# Separate features and target from each saved split
X_train = train_check.drop(columns=["Label", "Row_ID"])
y_train = train_check["Label"]

X_val = val_check.drop(columns=["Label", "Row_ID"])
y_val = val_check["Label"]

X_test = test_check.drop(columns=["Label", "Row_ID"])
y_test = test_check["Label"]

print("Training features:", X_train.shape)
print("Training target:", y_train.shape)

print("Validation features:", X_val.shape)
print("Validation target:", y_val.shape)

print("Test features:", X_test.shape)
print("Test target:", y_test.shape)

print("\nPASS: Features and targets separated successfully.")
print("\n=== CHECKING SPLIT FEATURES BEFORE PREPROCESSING ===")

print("\nMissing values:")
print("Training:", X_train.isna().sum().sum())
print("Validation:", X_val.isna().sum().sum())
print("Test:", X_test.isna().sum().sum())

print("\nInfinite values:")
print(
    "Training:",
    X_train.isin([float("inf"), float("-inf")]).sum().sum()
)
print(
    "Validation:",
    X_val.isin([float("inf"), float("-inf")]).sum().sum()
)
print(
    "Test:",
    X_test.isin([float("inf"), float("-inf")]).sum().sum()
)

print("\nData types:")
print(X_train.dtypes.value_counts())

if (
    X_train.isna().sum().sum() == 0
    and X_val.isna().sum().sum() == 0
    and X_test.isna().sum().sum() == 0
    and X_train.isin([float("inf"), float("-inf")]).sum().sum() == 0
    and X_val.isin([float("inf"), float("-inf")]).sum().sum() == 0
    and X_test.isin([float("inf"), float("-inf")]).sum().sum() == 0
):
    print("\nPASS: No missing or infinite values in any split.")
else:
    print("\nFAIL: Missing or infinite values detected.")
    print("\n=== CREATING LEAKAGE-SAFE PREPROCESSOR ===")

scaler = StandardScaler()

# Fit ONLY on training data
scaler.fit(X_train)

print("PASS: Scaler fitted using training data only.")
print("\n=== TRANSFORMING TRAIN / VALIDATION / TEST DATA ===")

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Scaled training shape:", X_train_scaled.shape)
print("Scaled validation shape:", X_val_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)

print("\nPASS: All three splits transformed successfully.")
print("\n=== VERIFYING SCALED FEATURES ===")

print("Training mean (first 5 features):")
print(X_train_scaled[:, :5].mean(axis=0))

print("\nTraining std (first 5 features):")
print(X_train_scaled[:, :5].std(axis=0))

print("\nValidation mean (first 5 features):")
print(X_val_scaled[:, :5].mean(axis=0))

print("\nTest mean (first 5 features):")
print(X_test_scaled[:, :5].mean(axis=0))

print("\nPASS: Scaled feature statistics checked.")
from sklearn.linear_model import LogisticRegression

print("\n=== PREPARING BASELINE MODEL ===")

baseline_model = LogisticRegression(
    max_iter=100,
    solver="saga",
    n_jobs=-1
)

print("Baseline model: Logistic Regression")
print("Solver: saga")
print("Maximum iterations: 100")
print("PASS: Baseline model created successfully.")
print("\n=== TRAINING BASELINE MODEL ===")

baseline_model.fit(X_train_scaled, y_train)

print("PASS: Baseline Logistic Regression trained successfully.")
print("\n=== EVALUATING BASELINE MODEL ON VALIDATION SET ===")

y_val_pred = baseline_model.predict(X_val_scaled)

print("Validation predictions generated.")
print("Number of validation predictions:", len(y_val_pred))

print("\nPASS: Validation prediction completed successfully.")
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

print("\n=== BASELINE VALIDATION METRICS ===")

accuracy = accuracy_score(y_val, y_val_pred)
balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

print("Validation Accuracy:", accuracy)
print("Validation Balanced Accuracy:", balanced_accuracy)
print("Validation Macro F1:", macro_f1)
print("Validation Weighted F1:", weighted_f1)

print("\nPASS: Baseline validation metrics calculated successfully.")
from sklearn.metrics import classification_report

print("\n=== BASELINE CLASSIFICATION REPORT ===")

report = classification_report(
    y_val,
    y_val_pred,
    zero_division=0
)

print(report)

print("PASS: Per-class validation report generated successfully.")
from sklearn.metrics import confusion_matrix

print("\n=== BASELINE CONFUSION MATRIX ===")

labels = sorted(y_val.unique())

cm = confusion_matrix(
    y_val,
    y_val_pred,
    labels=labels
)

print("Class labels:")
print(labels)

print("\nConfusion matrix:")
print(cm)

print("\nPASS: Validation confusion matrix generated successfully.")
print("\n=== EVALUATING BASELINE MODEL ON TEST SET ===")

y_test_pred = baseline_model.predict(X_test_scaled)

print("Test predictions generated.")
print("Number of test predictions:", len(y_test_pred))

print("\n=== BASELINE TEST METRICS ===")

test_accuracy = accuracy_score(y_test, y_test_pred)
test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
test_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
test_weighted_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

print("Test Accuracy:", test_accuracy)
print("Test Balanced Accuracy:", test_balanced_accuracy)
print("Test Macro F1:", test_macro_f1)
print("Test Weighted F1:", test_weighted_f1)

print("\nPASS: Baseline test metrics calculated successfully.")
print("\n=== BASELINE TEST CLASSIFICATION REPORT ===")

test_report = classification_report(
    y_test,
    y_test_pred,
    zero_division=0
)

print(test_report)

print("PASS: Per-class test report generated successfully.")
print("\n=== SAVING BASELINE RESULTS ===")

baseline_results = pd.DataFrame({
    "Dataset": ["Validation", "Test"],
    "Accuracy": [accuracy, test_accuracy],
    "Balanced_Accuracy": [balanced_accuracy, test_balanced_accuracy],
    "Macro_F1": [macro_f1, test_macro_f1],
    "Weighted_F1": [weighted_f1, test_weighted_f1]
})

print(baseline_results)

os.makedirs("results", exist_ok=True)

baseline_results.to_csv(
    "results/baseline_logistic_regression_results.csv",
    index=False
)

print("\nSaved:")
print("results/baseline_logistic_regression_results.csv")

print("\nPASS: Baseline results saved successfully.")
print("\n=== PREPARING IMBALANCE-AWARE MODEL ===")

balanced_model = LogisticRegression(
    max_iter=100,
    solver="saga",
    class_weight="balanced"
)

print("Model: Logistic Regression")
print("Class weighting: balanced")
print("Solver: saga")
print("Maximum iterations: 100")
print("PASS: Imbalance-aware model created successfully.")
print("\n=== TRAINING IMBALANCE-AWARE MODEL ===")

balanced_model.fit(X_train_scaled, y_train)

print("PASS: Imbalance-aware Logistic Regression trained successfully.")
print("\n=== EVALUATING IMBALANCE-AWARE MODEL ON VALIDATION SET ===")

y_val_balanced_pred = balanced_model.predict(X_val_scaled)

print("Validation predictions generated.")
print("Number of validation predictions:", len(y_val_balanced_pred))

balanced_val_accuracy = accuracy_score(
    y_val,
    y_val_balanced_pred
)

balanced_val_balanced_accuracy = balanced_accuracy_score(
    y_val,
    y_val_balanced_pred
)

balanced_val_macro_f1 = f1_score(
    y_val,
    y_val_balanced_pred,
    average="macro",
    zero_division=0
)

balanced_val_weighted_f1 = f1_score(
    y_val,
    y_val_balanced_pred,
    average="weighted",
    zero_division=0
)

print("\n=== IMBALANCE-AWARE VALIDATION METRICS ===")
print("Validation Accuracy:", balanced_val_accuracy)
print("Validation Balanced Accuracy:", balanced_val_balanced_accuracy)
print("Validation Macro F1:", balanced_val_macro_f1)
print("Validation Weighted F1:", balanced_val_weighted_f1)

print("\nPASS: Imbalance-aware validation metrics calculated successfully.")
print("\n=== IMBALANCE-AWARE VALIDATION CLASSIFICATION REPORT ===")

balanced_val_report = classification_report(
    y_val,
    y_val_balanced_pred,
    zero_division=0
)

print(balanced_val_report)

print("PASS: Imbalance-aware validation report generated successfully.")
print("\n=== EVALUATING IMBALANCE-AWARE MODEL ON TEST SET ===")

y_test_balanced_pred = balanced_model.predict(X_test_scaled)

print("Test predictions generated.")
print("Number of test predictions:", len(y_test_balanced_pred))

balanced_test_accuracy = accuracy_score(
    y_test,
    y_test_balanced_pred
)

balanced_test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_test_balanced_pred
)

balanced_test_macro_f1 = f1_score(
    y_test,
    y_test_balanced_pred,
    average="macro",
    zero_division=0
)

balanced_test_weighted_f1 = f1_score(
    y_test,
    y_test_balanced_pred,
    average="weighted",
    zero_division=0
)

print("\n=== IMBALANCE-AWARE TEST METRICS ===")
print("Test Accuracy:", balanced_test_accuracy)
print("Test Balanced Accuracy:", balanced_test_balanced_accuracy)
print("Test Macro F1:", balanced_test_macro_f1)
print("Test Weighted F1:", balanced_test_weighted_f1)

print("\nPASS: Imbalance-aware test metrics calculated successfully.")
print("\n=== IMBALANCE-AWARE TEST CLASSIFICATION REPORT ===")

balanced_test_report = classification_report(
    y_test,
    y_test_balanced_pred,
    zero_division=0
)

print(balanced_test_report)

print("PASS: Imbalance-aware test report generated successfully.")
print("\n=== SAVING IMBALANCE-AWARE RESULTS ===")

balanced_results = pd.DataFrame({
    "Dataset": ["Validation", "Test"],
    "Accuracy": [
        balanced_val_accuracy,
        balanced_test_accuracy
    ],
    "Balanced_Accuracy": [
        balanced_val_balanced_accuracy,
        balanced_test_balanced_accuracy
    ],
    "Macro_F1": [
        balanced_val_macro_f1,
        balanced_test_macro_f1
    ],
    "Weighted_F1": [
        balanced_val_weighted_f1,
        balanced_test_weighted_f1
    ]
})

print(balanced_results)

balanced_results.to_csv(
    "results/balanced_logistic_regression_results.csv",
    index=False
)

print("\nSaved:")
print("results/balanced_logistic_regression_results.csv")

print("PASS: Imbalance-aware results saved successfully.")
# Store baseline test metrics for comparison
baseline_test_accuracy = accuracy_score(
    y_test,
    baseline_model.predict(X_test_scaled)
)

baseline_test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    baseline_model.predict(X_test_scaled)
)

baseline_test_macro_f1 = f1_score(
    y_test,
    baseline_model.predict(X_test_scaled),
    average="macro",
    zero_division=0
)

baseline_test_weighted_f1 = f1_score(
    y_test,
    baseline_model.predict(X_test_scaled),
    average="weighted",
    zero_division=0
)
print("\n=== COMPARING BASELINE VS IMBALANCE-AWARE MODEL ===")

comparison = pd.DataFrame({
    "Model": [
        "Baseline Logistic Regression",
        "Imbalance-Aware Logistic Regression"
    ],
    "Test Accuracy": [
        baseline_test_accuracy,
        balanced_test_accuracy
    ],
    "Test Balanced Accuracy": [
        baseline_test_balanced_accuracy,
        balanced_test_balanced_accuracy
    ],
    "Test Macro F1": [
        baseline_test_macro_f1,
        balanced_test_macro_f1
    ],
    "Test Weighted F1": [
        baseline_test_weighted_f1,
        balanced_test_weighted_f1
    ]
})

print(comparison)

print("\n=== METRIC CHANGES ===")

print(
    "Accuracy change:",
    balanced_test_accuracy - baseline_test_accuracy
)

print(
    "Balanced Accuracy change:",
    balanced_test_balanced_accuracy - baseline_test_balanced_accuracy
)

print(
    "Macro F1 change:",
    balanced_test_macro_f1 - baseline_test_macro_f1
)

print(
    "Weighted F1 change:",
    balanced_test_weighted_f1 - baseline_test_weighted_f1
)

comparison.to_csv(
    "results/logistic_regression_comparison.csv",
    index=False
)

print("\nSaved:")
print("results/logistic_regression_comparison.csv")

print("PASS: Baseline vs imbalance-aware comparison completed successfully.")
print("\n=== PREPARING RANDOM FOREST MODEL ===")

from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

print("Model: Random Forest")
print("Number of trees: 100")
print("Class weighting: balanced_subsample")
print("Random state: 42")
print("PASS: Random Forest model created successfully.")
print("\n=== TRAINING RANDOM FOREST MODEL ===")

random_forest_model.fit(X_train, y_train)

print("PASS: Random Forest model trained successfully.")
print("\n=== EVALUATING RANDOM FOREST ON VALIDATION SET ===")

rf_val_pred = random_forest_model.predict(X_val)

rf_val_accuracy = accuracy_score(
    y_val,
    rf_val_pred
)

rf_val_balanced_accuracy = balanced_accuracy_score(
    y_val,
    rf_val_pred
)

rf_val_macro_f1 = f1_score(
    y_val,
    rf_val_pred,
    average="macro",
    zero_division=0
)

rf_val_weighted_f1 = f1_score(
    y_val,
    rf_val_pred,
    average="weighted",
    zero_division=0
)

print("\n=== RANDOM FOREST VALIDATION METRICS ===")
print("Validation Accuracy:", rf_val_accuracy)
print("Validation Balanced Accuracy:", rf_val_balanced_accuracy)
print("Validation Macro F1:", rf_val_macro_f1)
print("Validation Weighted F1:", rf_val_weighted_f1)

print("PASS: Random Forest validation metrics calculated successfully.")
print("\n=== EVALUATING RANDOM FOREST ON TEST SET ===")

rf_test_pred = random_forest_model.predict(X_test)

rf_test_accuracy = accuracy_score(
    y_test,
    rf_test_pred
)

rf_test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    rf_test_pred
)

rf_test_macro_f1 = f1_score(
    y_test,
    rf_test_pred,
    average="macro",
    zero_division=0
)

rf_test_weighted_f1 = f1_score(
    y_test,
    rf_test_pred,
    average="weighted",
    zero_division=0
)

print("\n=== RANDOM FOREST TEST METRICS ===")
print("Test Accuracy:", rf_test_accuracy)
print("Test Balanced Accuracy:", rf_test_balanced_accuracy)
print("Test Macro F1:", rf_test_macro_f1)
print("Test Weighted F1:", rf_test_weighted_f1)

print("PASS: Random Forest test metrics calculated successfully.")
print("\n=== RANDOM FOREST TEST CLASSIFICATION REPORT ===")

rf_test_report = classification_report(
    y_test,
    rf_test_pred,
    zero_division=0
)

print(rf_test_report)

print("PASS: Random Forest test classification report generated successfully.")
print("\n=== SAVING RANDOM FOREST RESULTS ===")

rf_results = pd.DataFrame({
    "Dataset": ["Validation", "Test"],
    "Accuracy": [rf_val_accuracy, rf_test_accuracy],
    "Balanced_Accuracy": [rf_val_balanced_accuracy, rf_test_balanced_accuracy],
    "Macro_F1": [rf_val_macro_f1, rf_test_macro_f1],
    "Weighted_F1": [rf_val_weighted_f1, rf_test_weighted_f1]
})

rf_results.to_csv(
    "results/random_forest_results.csv",
    index=False
)

print(rf_results)

print("\nSaved:")
print("results/random_forest_results.csv")

print("PASS: Random Forest results saved successfully.")
print("\n=== COMPARING ALL THREE MODELS ===")

three_model_comparison = pd.DataFrame({
    "Model": [
        "Baseline Logistic Regression",
        "Imbalance-Aware Logistic Regression",
        "Random Forest"
    ],
    "Test Accuracy": [
        test_accuracy,
        balanced_test_accuracy,
        rf_test_accuracy
    ],
    "Test Balanced Accuracy": [
        test_balanced_accuracy,
        balanced_test_balanced_accuracy,
        rf_test_balanced_accuracy
    ],
    "Test Macro F1": [
        test_macro_f1,
        balanced_test_macro_f1,
        rf_test_macro_f1
    ],
    "Test Weighted F1": [
        test_weighted_f1,
        balanced_test_weighted_f1,
        rf_test_weighted_f1
    ]
})

print(three_model_comparison)

three_model_comparison.to_csv(
    "results/three_model_comparison.csv",
    index=False
)

print("\nSaved:")
print("results/three_model_comparison.csv")

print("PASS: Three-model comparison completed successfully.")
print("\n=== RANDOM FOREST CONFUSION MATRIX ===")

from sklearn.metrics import confusion_matrix

rf_confusion = confusion_matrix(
    y_test,
    rf_test_pred,
    labels=random_forest_model.classes_
)

print("Class labels:")
print(list(random_forest_model.classes_))

print("\nConfusion matrix:")
print(rf_confusion)

print("\nPASS: Random Forest confusion matrix generated successfully.")
print("\n=== RANDOM FOREST PER-CLASS ANALYSIS ===")

rf_test_report_dict = classification_report(
    y_test,
    rf_test_pred,
    output_dict=True,
    zero_division=0
)

rf_per_class = pd.DataFrame(rf_test_report_dict).T

# Keep only actual attack/traffic classes
rf_per_class = rf_per_class.loc[
    random_forest_model.classes_
]

rf_per_class = rf_per_class[
    ["precision", "recall", "f1-score", "support"]
]

print(rf_per_class)

print("\n=== RANDOM FOREST LOW-RECALL CLASSES ===")

low_recall_classes = rf_per_class[
    rf_per_class["recall"] < 0.80
]

print(low_recall_classes)

print("\nPASS: Random Forest per-class analysis completed successfully.")
print("\n=== RANDOM FOREST SANITY CHECK ===")

# Check for duplicate feature rows between training and test sets
print("\nChecking for identical feature rows between training and test sets...")

train_feature_duplicates = pd.util.hash_pandas_object(
    X_train,
    index=False
)

test_feature_duplicates = pd.util.hash_pandas_object(
    X_test,
    index=False
)

train_hashes = set(train_feature_duplicates)
test_hashes = set(test_feature_duplicates)

cross_split_duplicate_hashes = train_hashes.intersection(test_hashes)

print("Unique training feature hashes:", len(train_hashes))
print("Unique test feature hashes:", len(test_hashes))
print("Feature hashes shared between train and test:",
      len(cross_split_duplicate_hashes))

if len(cross_split_duplicate_hashes) == 0:
    print("PASS: No identical feature rows detected between training and test sets.")
else:
    print("WARNING: Some identical feature rows exist between training and test sets.")

print("\nPASS: Random Forest sanity check completed successfully.")