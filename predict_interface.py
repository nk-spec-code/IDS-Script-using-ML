import joblib
import numpy as np

# Load trained model + features
rf = joblib.load("rf_model.pkl")
feature_names = joblib.load("feature_names.pkl")

def predict_attack(user_input_dict):
    """
    user_input_dict: dictionary of feature_name:value
    Example: {"duration": 10, "src_bytes": 200, "dst_bytes": 150, ...}
    """
    # Ensure correct feature order
    input_data = [user_input_dict.get(feat, 0) for feat in feature_names]
    input_array = np.array(input_data).reshape(1, -1)

    prediction = rf.predict(input_array)[0]

    if prediction == 0:
        print("✅ Normal Traffic Detected")
    else:
        print("🚨 Attack Detected!")

# --- Example run ---
sample_input = {feat: 0 for feat in feature_names}  # dummy data
sample_input[feature_names[0]] = 10  # set first feature to 10 for testing
predict_attack(sample_input)
