import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import zipfile
import os

# Nonaktifkan GPU pada TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Ekstrak model ZIP jika belum diekstrak
if not os.path.exists('saved_model_1/eye_diseases_model_1'):
    with zipfile.ZipFile('eye_saved_model_1.zip', 'r') as zip_ref:
        zip_ref.extractall()

# Load model
model_dir = 'saved_model_1/eye_diseases_model_1'
model = tf.saved_model.load(model_dir)

# Load labels
labels_file = 'saved_model_1/tflite_model/label.txt'
if os.path.exists(labels_file):
    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f]
else:
    labels = ["Label tidak ditemukan"]

# Fungsi prediksi
def predict(image):
    # Resize image dan normalisasi
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Lakukan prediksi
    predictions = model.signatures["serving_default"](tf.convert_to_tensor(img_array, dtype=tf.float32))
    output_key = list(predictions.keys())[0]  # Ambil key pertama dari output
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
