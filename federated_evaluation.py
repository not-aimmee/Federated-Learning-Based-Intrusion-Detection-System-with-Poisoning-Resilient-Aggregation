import flwr as fl
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

print("Flower version:", fl.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

print("\n=== FEDERATED LEARNING EVALUATION ===")
print("Flower environment initialized successfully.")
print("\n=== LOADING GROUP-AWARE SPLITS ===")

train_group = pd.read_csv(
    "data/splits/train_group_aware.csv"
)

val_group = pd.read_csv(
    "data/splits/validation_group_aware.csv"
)

test_group = pd.read_csv(
    "data/splits/test_group_aware.csv"
)

print("Training shape:", train_group.shape)
print("Validation shape:", val_group.shape)
print("Test shape:", test_group.shape)

print("\nLabels:")
print("Training:", train_group["Label"].nunique())
print("Validation:", val_group["Label"].nunique())
print("Test:", test_group["Label"].nunique())

print("\nPASS: Group-aware splits loaded successfully.")
X_train_fed = train_group.drop(columns=["Label", "Row_ID"])
y_train_fed = train_group["Label"]

X_val_fed = val_group.drop(columns=["Label", "Row_ID"])
y_val_fed = val_group["Label"]

X_test_fed = test_group.drop(columns=["Label", "Row_ID"])
y_test_fed = test_group["Label"]
print("\n=== CREATING FEDERATED CLIENT PARTITIONS ===")


# Number of simulated federated clients
NUM_CLIENTS = 5

# Use a fixed seed for reproducibility
rng = np.random.default_rng(42)

# Shuffle row positions
indices = np.arange(len(X_train_fed))
rng.shuffle(indices)

# Split training data among clients
client_indices = np.array_split(indices, NUM_CLIENTS)

client_data = []
from sklearn.preprocessing import StandardScaler

print("\n=== FITTING FEDERATED FEATURE SCALER ===")

scaler_fed = StandardScaler()

X_train_fed_scaled = scaler_fed.fit_transform(X_train_fed)
X_val_fed_scaled = scaler_fed.transform(X_val_fed)
X_test_fed_scaled = scaler_fed.transform(X_test_fed)

print("Scaled training shape:", X_train_fed_scaled.shape)
print("Scaled validation shape:", X_val_fed_scaled.shape)
print("Scaled test shape:", X_test_fed_scaled.shape)

print("\nPASS: Federated scaler fitted on training data only.")
for client_id, idx in enumerate(client_indices, start=1):

    X_client = X_train_fed_scaled[idx]
    y_client = y_train_fed.iloc[idx]

    client_data.append(
        (X_client, y_client)
    )

    print(
        f"Client {client_id}: "
        f"{len(X_client)} samples, "
        f"{y_client.nunique()} classes"
    )

print("\nTotal client samples:")
print(
    sum(len(X_client) for X_client, _ in client_data)
)

print("\nPASS: Federated client partitions created.")
print("\n=== VERIFYING FEDERATED CLIENT PARTITIONS ===")

# Collect all client indices
all_client_indices = np.concatenate(client_indices)

print("Total assigned rows:", len(all_client_indices))
print("Unique assigned rows:", len(np.unique(all_client_indices)))
print("Expected training rows:", len(X_train_fed))

# Check whether every training row was assigned exactly once
if (
    len(all_client_indices) == len(X_train_fed)
    and len(np.unique(all_client_indices)) == len(X_train_fed)
):
    print("PASS: Every training row belongs to exactly one client.")
else:
    print("FAIL: Client partition overlap or missing rows detected.")

# Check client sizes
print("\nClient sizes:")
for client_id, idx in enumerate(client_indices, start=1):
    print(f"Client {client_id}: {len(idx)} rows")

print("\nPASS: Federated partition verification completed.")
print("\n=== PREPARING FEDERATED EVALUATION DATA ===")

print("Validation features:", X_val_fed.shape)
print("Validation labels:", y_val_fed.shape)

print("Test features:", X_test_fed.shape)
print("Test labels:", y_test_fed.shape)

print("\nValidation classes:", y_val_fed.nunique())
print("Test classes:", y_test_fed.nunique())

# Confirm feature alignment
if list(X_train_fed.columns) == list(X_val_fed.columns) == list(X_test_fed.columns):
    print("\nPASS: Feature columns are aligned across train/validation/test.")
else:
    print("\nFAIL: Feature columns are not aligned.")

print("\nPASS: Federated evaluation data prepared.")
print("\n=== DEFINING FEDERATED LOGISTIC REGRESSION ===")

from sklearn.linear_model import LogisticRegression

NUM_CLASSES = len(np.unique(y_train_fed))
NUM_FEATURES = X_train_fed.shape[1]

print("Number of features:", NUM_FEATURES)
print("Number of classes:", NUM_CLASSES)

print("\nPASS: Federated model configuration prepared.")
print("\n=== DEFINING FLOWER CLIENT ===")

from flwr.client import NumPyClient


class FederatedClient(NumPyClient):

    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.model = LogisticRegression(
            max_iter=100,
            solver="lbfgs",
        )

    def get_parameters(self, config):
        if not hasattr(self.model, "coef_"):
            return [
                np.zeros(
                    (NUM_CLASSES, NUM_FEATURES),
                    dtype=np.float64
                ),
                np.zeros(
                    NUM_CLASSES,
                    dtype=np.float64
                )
            ]

        return [
            self.model.coef_,
            self.model.intercept_
        ]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

        self.model.n_features_in_ = NUM_FEATURES

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        self.model.fit(
            self.X,
            self.y
        )

        return (
            self.get_parameters(config),
            len(self.X),
            {}
        )

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        predictions = self.model.predict(
            X_val_fed
        )

        accuracy = accuracy_score(
            y_val_fed,
            predictions
        )

        balanced_accuracy = balanced_accuracy_score(
            y_val_fed,
            predictions
        )

        macro_f1 = f1_score(
            y_val_fed,
            predictions,
            average="macro",
            zero_division=0
        )

        return (
            float(accuracy),
            len(X_val_fed),
            {
                "balanced_accuracy": float(
                    balanced_accuracy
                ),
                "macro_f1": float(
                    macro_f1
                )
            }
        )


print("PASS: Flower client defined successfully.")
print("\n=== CREATING INITIAL GLOBAL MODEL ===")

# Use the actual string labels from the federated training data.
GLOBAL_CLASSES = np.sort(y_train_fed.unique())

print("Global classes:", GLOBAL_CLASSES)
print("Number of global classes:", len(GLOBAL_CLASSES))

# Initialize model with the correct number of classes and features.
global_model = LogisticRegression(
    max_iter=100,
    solver="lbfgs"
)

# Store the global parameter arrays.
initial_parameters = [
    np.zeros(
        (len(GLOBAL_CLASSES), NUM_FEATURES),
        dtype=np.float64
    ),
    np.zeros(
        len(GLOBAL_CLASSES),
        dtype=np.float64
    )
]

print("Initial coefficient shape:", initial_parameters[0].shape)
print("Initial intercept shape:", initial_parameters[1].shape)

print("\nPASS: Initial global model created.")
print("\n=== CHECKING CLIENT CLASS COVERAGE ===")

for client_id, (X_client, y_client) in enumerate(
    client_data,
    start=1
):
    client_classes = np.sort(y_client.unique())

    missing_classes = [
        cls for cls in GLOBAL_CLASSES
        if cls not in client_classes
    ]

    print(
        f"Client {client_id}: "
        f"{len(client_classes)} local classes, "
        f"{len(missing_classes)} missing global classes"
    )

    if missing_classes:
        print("  Missing:", missing_classes)

print("\nPASS: Client class coverage checked.")
print("\n=== DEFINING CLASS-ALIGNED PARAMETER FUNCTION ===")

def align_client_parameters(model, global_classes):
    """
    Align a locally trained Logistic Regression model
    to the fixed global class order.
    """

    aligned_coef = np.zeros(
        (len(global_classes), NUM_FEATURES),
        dtype=np.float64
    )

    aligned_intercept = np.zeros(
        len(global_classes),
        dtype=np.float64
    )

    for local_index, class_name in enumerate(model.classes_):
        global_index = np.where(
            global_classes == class_name
        )[0][0]

        aligned_coef[global_index] = model.coef_[local_index]
        aligned_intercept[global_index] = model.intercept_[local_index]

    return [
        aligned_coef,
        aligned_intercept
    ]

print("\n=== RUNNING FEDERATED TRAINING ===")

NUM_ROUNDS = 3

global_parameters = initial_parameters

for round_number in range(1, NUM_ROUNDS + 1):

    print(f"\n--- Federated Round {round_number} ---")

    client_parameters = []
    client_sizes = []

    for client_id, (X_client, y_client) in enumerate(
        client_data,
        start=1
    ):

        local_model = LogisticRegression(
            max_iter=100,
            solver="lbfgs"
        )

        local_model.fit(
            X_client,
            y_client
        )

        aligned_parameters = align_client_parameters(
            local_model,
            GLOBAL_CLASSES
        )

        client_parameters.append(aligned_parameters)
        client_sizes.append(len(X_client))

        print(
            f"Client {client_id}: "
            f"trained on {len(X_client)} samples"
        )

    total_samples = sum(client_sizes)

    global_coef = sum(
        parameters[0] * (size / total_samples)
        for parameters, size
        in zip(client_parameters, client_sizes)
    )

    global_intercept = sum(
        parameters[1] * (size / total_samples)
        for parameters, size
        in zip(client_parameters, client_sizes)
    )

    global_parameters = [
        global_coef,
        global_intercept
    ]

    print(
        "Global coefficient shape:",
        global_parameters[0].shape
    )

    print(
        "Global intercept shape:",
        global_parameters[1].shape
    )

print("\nPASS: Federated training completed.")
print("\n=== EVALUATING GLOBAL FEDERATED MODEL ===")

global_model = LogisticRegression(
    max_iter=100,
    solver="lbfgs"
)

global_model.classes_ = GLOBAL_CLASSES
global_model.coef_ = global_parameters[0]
global_model.intercept_ = global_parameters[1]
global_model.n_features_in_ = NUM_FEATURES

# Validation predictions
val_predictions = global_model.predict(
    X_val_fed_scaled
)

val_accuracy = accuracy_score(
    y_val_fed,
    val_predictions
)

val_balanced_accuracy = balanced_accuracy_score(
    y_val_fed,
    val_predictions
)

val_macro_f1 = f1_score(
    y_val_fed,
    val_predictions,
    average="macro",
    zero_division=0
)

# Test predictions
test_predictions = global_model.predict(
    X_test_fed_scaled
)

test_accuracy = accuracy_score(
    y_test_fed,
    test_predictions
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test_fed,
    test_predictions
)

test_macro_f1 = f1_score(
    y_test_fed,
    test_predictions,
    average="macro",
    zero_division=0
)

print("\nFederated Validation Results:")
print("Accuracy:", val_accuracy)
print("Balanced Accuracy:", val_balanced_accuracy)
print("Macro F1:", val_macro_f1)

print("\nFederated Test Results:")
print("Accuracy:", test_accuracy)
print("Balanced Accuracy:", test_balanced_accuracy)
print("Macro F1:", test_macro_f1)

print("\nPASS: Global federated model evaluation completed.")
import pandas as pd

print("\n=== SAVING FEDERATED RESULTS ===")

federated_results = pd.DataFrame([
    {
        "Model": "Federated Logistic Regression",
        "Split": "Validation",
        "Accuracy": val_accuracy,
        "Balanced Accuracy": val_balanced_accuracy,
        "Macro F1": val_macro_f1
    },
    {
        "Model": "Federated Logistic Regression",
        "Split": "Test",
        "Accuracy": test_accuracy,
        "Balanced Accuracy": test_balanced_accuracy,
        "Macro F1": test_macro_f1
    }
])

federated_results.to_csv(
    "federated_results.csv",
    index=False
)

print("\nFederated results:")
print(federated_results)

print("\nSaved to: federated_results.csv")
print("\nPASS: Federated results saved successfully.")