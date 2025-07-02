import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
import zipfile

# Nonaktifkan GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# URL model yang benar di Hugging Face
MODEL_URL = "https://huggingface.co/harrylbi/model.tflite/resolve/main/eye_saved_model_1.zip"
MODEL_ZIP_PATH = "eye_saved_model_1.zip"
MODEL_DIR = "saved_model_1/eye_diseases_model_1"
LABEL_PATH = "saved_model_1/tflite_model/label.txt"

# Download dan ekstrak model jika belum ada
if not os.path.exists(MODEL_DIR):
    try:
        print("Mengunduh model dari Hugging Face...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_ZIP_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall()
    except Exception as e:
        print(f"Gagal mengunduh atau mengekstrak model: {e}")
        exit(1)

# Load model TensorFlow SavedModel
model = tf.saved_model.load(MODEL_DIR)

# Load label
if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, 'r') as f:
        labels = [line.strip() for line in f]
else:
    labels = ["Label tidak ditemukan"]

# Fungsi prediksi
def predict(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.signatures["serving_default"](tf.convert_to_tensor(img_array, dtype=tf.float32))
    output_key = list(predictions.keys())[0]
    predicted_probabilities = predictions[output_key].numpy()
    predicted_index = np.argmax(predicted_probabilities)

    return f"Prediksi: {labels[predicted_index]}"

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Hasil Prediksi"),
    title="Prediksi Penyakit Mata dengan AI"
)

if __name__ == "__main__":
    interface.launch()
