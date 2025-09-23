# bandgap_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition


# ---------------------------
# 1. Load Dataset
# ---------------------------
def load_data(csv_path):
    """Load dataset from CSV"""
    df = pd.read_csv(csv_path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------
# 2. Feature Engineering
# ---------------------------
def featurize(df, formula_column='formula', target_column='bandgap'):
    """Convert chemical formulas into features using Matminer"""
    df = df.dropna(subset=[formula_column, target_column])
    ep_feat = ElementProperty.from_preset(preset_name="magpie")

    # Generate features
    features = []
    for formula in df[formula_column]:
        comp = Composition(formula)
        feats = ep_feat.featurize(comp)
        features.append(feats)

    X = pd.DataFrame(features)
    y = df[target_column].values
    print(f"Features created: {X.shape[1]} features")
    return X, y


# ---------------------------
# 3. Train Model
# ---------------------------
def train_model(X, y):
    """Train a simple Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")

    return model


# ---------------------------
# 4. Main Script
# ---------------------------
if __name__ == "__main__":
    # Replace with your CSV path
    csv_path = "sample_bandgap_data.csv"

    # Load, featurize, and train
    df = load_data(csv_path)
    X, y = featurize(df)
    model = train_model(X, y)
