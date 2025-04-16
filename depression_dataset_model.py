import pandas as pd
import joblib
import wandb
import traceback
import numpy as np
import time
# No separate import needed for Settings if using wandb.Settings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix # Keep confusion_matrix import for metric calculation
)
from sklearn.impute import SimpleImputer

# --- Configuration ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3
IMPUTATION_STRATEGY = 'mean'
DATASET_PATH = "/home/nasir-hussain/Pictures/Depression_Detection_Using_Machine_Learning/dataset/depressionDataset.csv"
WANDB_PROJECT = "ml-classification-pipeline"
WANDB_RUN_NAME = "gridsearch_model_tuning_minimal_charts" # Updated name
WANDB_API_KEY = "bddd5c61e22807e3984a44eb48c934da052d1cd4" # Consider using environment variables or CLI login


# --- Wandb Setup ---
try:
    wandb.login(key=WANDB_API_KEY)
except Exception as e:
    print(f"Wandb login failed (maybe already logged in?): {e}")

# Define settings to disable default system stats logging
wandb_settings = wandb.Settings(_disable_stats=True)

run = wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "imputation_strategy": IMPUTATION_STRATEGY,
        "dataset_path": DATASET_PATH,
    },
    settings=wandb_settings
)
config = wandb.config

# --- Data Loading and Preprocessing ---
print("🔄 Loading and preprocessing data...")
df = pd.read_csv(config.dataset_path)
wandb.log({"initial_rows": df.shape[0], "initial_cols": df.shape[1]})

df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['start.time'] = pd.to_datetime(df['start.time'], errors='coerce')
df['hour'] = df['time'].dt.hour
df['dayofweek'] = df['time'].dt.dayofweek

period_encoder = LabelEncoder()
df['period.name'] = period_encoder.fit_transform(df['period.name'].astype(str))

features = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'hour', 'dayofweek', 'period.name']
# Log features used after definition
config.features_used = features
X = df[features]
y = df['class']

initial_target_dist = y.value_counts(dropna=False).to_dict()
mask = y.notna()
X = X[mask]
y = y[mask]
wandb.log({
    "target_distribution_before_nan_drop": initial_target_dist,
    "rows_after_target_nan_drop": len(y),
    "target_distribution_after_nan_drop": y.value_counts().to_dict()
})

le_y = LabelEncoder()
y = le_y.fit_transform(y)
class_mapping = dict(zip(le_y.classes_, le_y.transform(le_y.classes_)))
wandb.log({"target_class_mapping": class_mapping})
class_names = [str(cls) for cls in le_y.classes_]

imputer = SimpleImputer(strategy=config.imputation_strategy)
X = imputer.fit_transform(X)
wandb.log({"processed_rows": X.shape[0], "processed_cols": X.shape[1]})

assert not np.isnan(X).any(), "Found NaNs in X after imputation"
assert np.isfinite(X).all(), "Found inf or -inf in X after imputation"
assert not np.isnan(y).any(), "Found NaNs in y after encoding"
assert np.isfinite(y).all(), "Found inf or -inf in y after encoding"

# --- Train-test split ---
print("🔪 Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.test_size, stratify=y, random_state=config.random_state
)
wandb.log({
    "train_set_size": len(X_train),
    "test_set_size": len(X_test)
})

# --- Model definitions ---
model_grid = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=config.random_state),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, random_state=config.random_state),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    }
}
# Log the hyperparameter grids being searched
wandb.config.update({"model_hyperparameter_grids": model_grid})


# --- Train & evaluate ---
best_model = None
best_score = 0
best_model_name = None
best_probas = None
best_preds = None

for name, config_dict in model_grid.items():
    print(f"\n🔧 Running GridSearchCV for {name}...")
    start_time = time.time()
    grid = GridSearchCV(config_dict['model'], config_dict['params'], cv=config.cv_folds, scoring='accuracy', n_jobs=-1, error_score='raise')

    try:
        grid.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        best_cv_params = grid.best_params_
        best_cv_score = grid.best_score_

        preds = grid.predict(X_test)
        probas = grid.predict_proba(X_test) if hasattr(grid, "predict_proba") else None

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        # cm = confusion_matrix(y_test, preds) # Calculate CM if needed for metrics, but don't plot here

        # Log metrics and info for THIS model to wandb (NO charts here)
        log_dict = {
            f"{name}_test_accuracy": acc,
            f"{name}_test_precision_weighted": prec,
            f"{name}_test_recall_weighted": rec,
            f"{name}_test_f1_weighted": f1,
            f"{name}_best_cv_score": best_cv_score,
            f"{name}_best_cv_params": best_cv_params,
            f"{name}_training_time_seconds": training_time,
            # ----> REMOVED per-model confusion matrix plot <----
            # f"{name}_confusion_matrix": wandb.plot.confusion_matrix(...)
        }
        # Add per-class metrics from classification report
        try:
            report_dict = classification_report(y_test, preds, target_names=class_names, output_dict=True)
            for class_or_avg, metrics in report_dict.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        log_dict[f"{name}_test_{class_or_avg}_{metric_name}"] = value
        except Exception as report_e:
            print(f"⚠️ Could not log classification report metrics for {name}: {report_e}")

        wandb.log(log_dict)

        print(f"    Best CV Params for {name}: {best_cv_params}")
        print(f"    Mean CV Accuracy for {name}: {best_cv_score:.4f}")
        print(f"    Test Accuracy for {name}: {acc:.4f}")
        print(f"    Training Time for {name}: {training_time:.2f} seconds")

        if acc > best_score:
            print(f"    🚀 New best model found: {name}")
            best_score = acc
            best_model = grid.best_estimator_
            best_model_name = name
            best_probas = probas
            best_preds = preds

    except Exception as e:
        end_time = time.time()
        print(f"❌ Error while training {name} after {end_time - start_time:.2f} seconds: {e}")
        traceback.print_exc()
        wandb.alert(title=f"Training failed for {name}", text=f"{traceback.format_exc()}")
        wandb.log({f"{name}_training_status": "failed", f"{name}_error_message": str(e)})


# --- Save best model and Log Final Results ---
if best_model:
    model_path = f"best_model_{best_model_name}.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n✅ Best overall model: {best_model_name} with test accuracy {best_score:.4f}")
    print(f"✅ Model saved to {model_path}")

    wandb.summary["best_model_name"] = best_model_name
    wandb.summary["best_model_test_accuracy"] = best_score
    wandb.summary["best_model_params"] = best_model.get_params()

    artifact = wandb.Artifact(f'{best_model_name}-model', type='model')
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    print(f"✅ Model artifact uploaded to wandb run.")

    # Log performance plots for the BEST model using wandb.sklearn
    # This bundles essential plots like Confusion Matrix, ROC, PR Curve, Feature Importance (if applicable)
    if best_probas is not None:
         try:
            print("📊 Logging sklearn summary plots for best model...")
            wandb.sklearn.plot_classifier(
                best_model, X_train, X_test, y_train, y_test,
                y_pred=best_preds,
                y_probas=best_probas,
                labels=class_names,
                model_name=f'BestModel_{best_model_name}',
                feature_names=features
            )
            print("✅ Sklearn summary plots logged to wandb.")
         except Exception as plot_e:
            print(f"⚠️ Could not log sklearn plots: {plot_e}")
            traceback.print_exc()

    # ----> REMOVED redundant manual feature importance plot <----
    # Log feature importances for the best model (if applicable)
    # if hasattr(best_model, 'feature_importances_'):
    #     try:
    #         importances = best_model.feature_importances_
    #         # ... (code to create dataframe) ...
    #         # wandb.log({"best_model_feature_importances": wandb.Table(...)})
    #         # wandb.log({"best_model_feature_importances_plot": wandb.plot.bar(...)})
    #         # print("✅ Feature importances logged to wandb.")
    #     except Exception as fi_e:
    #         print(f"⚠️ Could not log feature importances: {fi_e}")


    # Detailed classification report logging for BEST model (kept)
    try:
        final_report_dict = classification_report(y_test, best_preds, target_names=class_names, output_dict=True)
        detailed_report_log = {}
        for label, metrics in final_report_dict.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    detailed_report_log[f"best_model_{label}_{metric_name}"] = value
            else:
                 detailed_report_log[f"best_model_{label}"] = metrics
        wandb.log(detailed_report_log)
        print("✅ Detailed classification report for best model logged.")
    except Exception as e:
        print(f"⚠️ Could not log final classification report details: {e}")
        traceback.print_exc()

else:
    print("\n❌ No model trained successfully.")
    wandb.summary["status"] = "Failed - No model trained"

# --- Finish Wandb Run ---
print("\n🏁 Finishing wandb run...")
wandb.finish()
print("✨ Script finished.")