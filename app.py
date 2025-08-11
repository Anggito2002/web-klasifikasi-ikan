import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG19
import xgboost as xgb

app = Flask(__name__)

# Load VGG19
vgg_model = VGG19(weights='imagenet', include_top=False, pooling='avg')

# Load model XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("model_xgb.json")

# Load Label Encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def extract_features(img_path):
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    features = vgg_model.predict(image, verbose=0)
    return features

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            features = extract_features(filepath)

            pred_class_idx = xgb_model.predict(features)[0]
            pred_class_name = label_encoder.inverse_transform([pred_class_idx])[0]

            probas = xgb_model.predict_proba(features)[0]
            probas_dict = {
                label_encoder.inverse_transform([i])[0]: float(probas[i])
                for i in range(len(probas))
            }

            return render_template(
                "index.html",
                image_path=filepath,
                prediction=pred_class_name,
                probas=probas_dict
            )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
