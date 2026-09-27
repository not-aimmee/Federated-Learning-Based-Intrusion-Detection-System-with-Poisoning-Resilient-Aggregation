import pandas as pd

print("=== FINAL MODEL COMPARISON ===")

results = pd.DataFrame([
    {
        "Model": "Baseline Logistic Regression",
        "Split": "Test",
        "Accuracy": 0.941729,
        "Balanced Accuracy": 0.294528,
        "Macro F1": 0.309474
    },
    {
        "Model": "Balanced Logistic Regression",
        "Split": "Test",
        "Accuracy": 0.643396,
        "Balanced Accuracy": 0.830064,
        "Macro F1": 0.274074
    },
    {
        "Model": "Original Random Forest",
        "Split": "Test",
        "Accuracy": 0.998151,
        "Balanced Accuracy": 0.848884,
        "Macro F1": 0.871359
    },
    {
        "Model": "Group-Aware Logistic Regression",
        "Split": "Test",
        "Accuracy": 0.942844,
        "Balanced Accuracy": 0.296512,
        "Macro F1": 0.313158
    },
    {
        "Model": "Group-Aware Random Forest",
        "Split": "Test",
        "Accuracy": 0.998515,
        "Balanced Accuracy": 0.845963,
        "Macro F1": 0.869130
    },
    {
        "Model": "Federated Logistic Regression",
        "Split": "Test",
        "Accuracy": 0.9717688942453502,
        "Balanced Accuracy": 0.45177878013112155,
        "Macro F1": 0.4656539075469109
    }
])

results.to_csv(
    "final_model_comparison.csv",
    index=False
)

print("\nFinal comparison:")
print(results.to_string(index=False))

print("\nSaved to: final_model_comparison.csv")
print("\nPASS: Final model comparison created successfully.")