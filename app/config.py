import os

# Path to the trained model bundle (.joblib dict with "model"/"features"/"classes" keys).
# Point MODEL_PATH at a different location via an environment variable (e.g. on Render).
MODEL_PATH = os.environ.get("MODEL_PATH", "fish_rf_model.joblib")
