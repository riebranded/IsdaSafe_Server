import os

# Path to the trained RandomForestClassifier .joblib file.
# Drop your model file at model/fish_model.joblib, or point MODEL_PATH at a
# different location via an environment variable (e.g. on Render).
MODEL_PATH = os.environ.get("MODEL_PATH", "model/fish_model.joblib")

# Exact column order the model expects as input. Edit this if your model
# was trained with different feature names/order.
FEATURE_ORDER = ["ammonia", "dissolved_oxygen", "ph", "temperature"]
