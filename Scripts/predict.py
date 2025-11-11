import sys
from keras.models import load_model
from notebooks.preprocessing_utils import preprocess_images, TARGET_SIZE
from skimage.io import imread
import numpy as np
import custom_layers  # Ensure custom layers are registered

# --------- 1. Load trained model ---------
MODEL_PATH = "/mnt/g/which one is it/models/CNN_ViT_hybrid.keras"
model = load_model(MODEL_PATH)

# --------- 2. Class labels (ID → Name) ---------
class_names = {
    0: "Speed limit 20", 1: "Speed limit 30", 2: "Speed limit 50",
    3: "Speed limit 60", 4: "Speed limit 70", 5: "Speed limit 80",
    6: "Restriction ends 80", 7: "Speed limit 100", 8: "Speed limit 120",
    9: "No overtaking", 10: "No overtaking (trucks)", 11: "Priority at next intersection",
    12: "Priority road", 13: "Give way", 14: "Stop", 15: "No traffic both ways",
    16: "No trucks", 17: "No entry", 18: "Danger", 19: "Bend left",
    20: "Bend right", 21: "Bend", 22: "Uneven road", 23: "Slippery road",
    24: "Road narrows", 25: "Construction", 26: "Traffic signal",
    27: "Pedestrian crossing", 28: "School crossing", 29: "Cycles crossing",
    30: "Snow", 31: "Animals", 32: "Restriction ends", 33: "Go right",
    34: "Go left", 35: "Go straight", 36: "Go right or straight",
    37: "Go left or straight", 38: "Keep right", 39: "Keep left",
    40: "Roundabout", 41: "Restriction ends (overtaking)",
    42: "Restriction ends (overtaking trucks)",
}

# --------- 3. Preprocess input image ---------
def preprocess_single_image(image_path):
    img = imread(image_path)
    img_array = preprocess_images([img], TARGET_SIZE)
    return img_array  # shape (1,80,80,3), float32 [0,1]

# --------- 4. Prediction function ---------
def predict_image(image_path):
    img_array = preprocess_single_image(image_path)
    preds = model.predict(img_array)[0]     # shape (43,)
    
    # Top-3 indices
    top3_idx = preds.argsort()[-3:][::-1]
    
    print(f"\nPrediction for: {image_path}")
    for rank, idx in enumerate(top3_idx, start=1):
        print(f"{rank}. {class_names[idx]} – {preds[idx]*100:.2f}%")

# --------- 5. Main entry ---------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    predict_image(img_path)
