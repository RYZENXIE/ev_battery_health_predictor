"""
Training Pipeline for EV Battery Health & Range Predictor
Trains:
1. SOH (State of Health) Regressor
2. Real-World Driving Range Regressor
3. Battery Degradation Status Classifier
Evaluates performance metrics (R2, RMSE, MAE, Classification Report)
and exports trained pipelines using joblib.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, classification_report

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "ev_battery_telemetry.csv")
    models_dir = os.path.join(base_dir, "saved_models")
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print("Dataset not found. Generating dataset first...")
        from generate_dataset import generate_ev_dataset
        df = generate_ev_dataset(6000)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

    # -------------------------------------------------------------
    # 1. Feature Definitions
    # -------------------------------------------------------------
    # SOH Features (Historical & Battery physical wear features)
    soh_cat_features = ["battery_type"]
    soh_num_features = [
        "nominal_capacity_kwh", "charge_cycles", "fast_charge_ratio",
        "avg_operating_temp_c", "avg_dod_pct", "internal_resistance_mohm"
    ]
    soh_features = soh_cat_features + soh_num_features

    # Range Features (SOH + Driving conditions & dynamics)
    range_cat_features = ["battery_type", "driving_style"]
    range_num_features = [
        "nominal_capacity_kwh", "soh_pct", "ambient_temp_c",
        "avg_speed_kmh", "hvac_level", "elevation_gain_m",
        "payload_kg", "tire_pressure_psi", "internal_resistance_mohm"
    ]
    range_features = range_cat_features + range_num_features

    # -------------------------------------------------------------
    # 2. Preprocessor Pipelines
    # -------------------------------------------------------------
    soh_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), soh_cat_features),
            ("num", StandardScaler(), soh_num_features)
        ]
    )

    range_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), range_cat_features),
            ("num", StandardScaler(), range_num_features)
        ]
    )

    # -------------------------------------------------------------
    # 3. Train SOH Regressor
    # -------------------------------------------------------------
    print("\n--- [1/3] Training SOH (State of Health) Regressor ---")
    X_soh = df[soh_features]
    y_soh = df["soh_pct"]

    X_train_soh, X_test_soh, y_train_soh, y_test_soh = train_test_split(
        X_soh, y_soh, test_size=0.2, random_state=42
    )

    soh_pipeline = Pipeline([
        ("preprocessor", soh_preprocessor),
        ("regressor", GradientBoostingRegressor(n_estimators=180, learning_rate=0.08, max_depth=4, random_state=42))
    ])

    soh_pipeline.fit(X_train_soh, y_train_soh)
    y_pred_soh = soh_pipeline.predict(X_test_soh)

    soh_r2 = r2_score(y_test_soh, y_pred_soh)
    soh_rmse = np.sqrt(mean_squared_error(y_test_soh, y_pred_soh))
    soh_mae = mean_absolute_error(y_test_soh, y_pred_soh)

    print(f"SOH Model Metrics -> R2: {soh_r2:.4f} | RMSE: {soh_rmse:.3f}% | MAE: {soh_mae:.3f}%")

    # -------------------------------------------------------------
    # 4. Train Real-World Range Regressor
    # -------------------------------------------------------------
    print("\n--- [2/3] Training Real-World Driving Range Regressor ---")
    X_range = df[range_features]
    y_range = df["real_world_range_km"]

    X_train_range, X_test_range, y_train_range, y_test_range = train_test_split(
        X_range, y_range, test_size=0.2, random_state=42
    )

    range_pipeline = Pipeline([
        ("preprocessor", range_preprocessor),
        ("regressor", GradientBoostingRegressor(n_estimators=220, learning_rate=0.07, max_depth=5, random_state=42))
    ])

    range_pipeline.fit(X_train_range, y_train_range)
    y_pred_range = range_pipeline.predict(X_test_range)

    range_r2 = r2_score(y_test_range, y_pred_range)
    range_rmse = np.sqrt(mean_squared_error(y_test_range, y_pred_range))
    range_mae = mean_absolute_error(y_test_range, y_pred_range)

    print(f"Range Model Metrics -> R2: {range_r2:.4f} | RMSE: {range_rmse:.2f} km | MAE: {range_mae:.2f} km")

    # -------------------------------------------------------------
    # 5. Train Health Status Classifier
    # -------------------------------------------------------------
    print("\n--- [3/3] Training Battery Degradation Classifier ---")
    y_clf = df["battery_health_status"]
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_soh, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    clf_pipeline = Pipeline([
        ("preprocessor", soh_preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])

    clf_pipeline.fit(X_train_clf, y_train_clf)
    y_pred_clf = clf_pipeline.predict(X_test_clf)
    clf_acc = accuracy_score(y_test_clf, y_pred_clf)

    print(f"Health Status Accuracy: {clf_acc * 100:.2f}%")
    print(classification_report(y_test_clf, y_pred_clf))

    # -------------------------------------------------------------
    # 6. Feature Importance Extraction
    # -------------------------------------------------------------
    # Extract feature importance for Range model
    num_names = range_num_features
    # For one-hot encoded features
    ohe = range_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(range_cat_features))
    all_feature_names = cat_names + num_names
    importances = range_pipeline.named_steps["regressor"].feature_importances_

    feature_imp_dict = {name: float(imp) for name, imp in sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)}

    # -------------------------------------------------------------
    # 7. Save Models and Metadata
    # -------------------------------------------------------------
    soh_model_path = os.path.join(models_dir, "soh_predictor.joblib")
    range_model_path = os.path.join(models_dir, "range_predictor.joblib")
    clf_model_path = os.path.join(models_dir, "status_classifier.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")

    joblib.dump(soh_pipeline, soh_model_path)
    joblib.dump(range_pipeline, range_model_path)
    joblib.dump(clf_pipeline, clf_model_path)

    metrics_data = {
        "soh_model": {
            "r2_score": round(float(soh_r2), 4),
            "rmse": round(float(soh_rmse), 4),
            "mae": round(float(soh_mae), 4),
            "algorithm": "GradientBoostingRegressor (180 trees, depth 4)"
        },
        "range_model": {
            "r2_score": round(float(range_r2), 4),
            "rmse": round(float(range_rmse), 4),
            "mae": round(float(range_mae), 4),
            "algorithm": "GradientBoostingRegressor (220 trees, depth 5)",
            "feature_importance": feature_imp_dict
        },
        "classifier_model": {
            "accuracy": round(float(clf_acc), 4),
            "algorithm": "RandomForestClassifier (100 trees, depth 6)"
        }
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"\nAll models successfully trained and exported to: {models_dir}")
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
