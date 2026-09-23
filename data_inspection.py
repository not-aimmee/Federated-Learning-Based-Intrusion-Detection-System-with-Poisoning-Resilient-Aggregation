import pandas as pd

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
print("\n=== LABEL DISTRIBUTION OF ALL CICIDS2017 FILES ===")

import os

data_folder = "data"

for filename in sorted(os.listdir(data_folder)):

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

    if filename.endswith(".csv"):

        file_path = os.path.join(data_folder, filename)

        temp_df = pd.read_csv(file_path)

        label_counts = temp_df[" Label"].value_counts()

        for label, count in label_counts.items():
            all_label_counts[label] = all_label_counts.get(label, 0) + count

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