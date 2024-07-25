from flask import Flask, render_template, request, url_for
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import joblib

app = Flask(__name__)

# Membuat direktori untuk mengunggah gambar jika belum ada
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Memuat model MobileNetV2 langsung dari TensorFlow
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Menambahkan lapisan Global Average Pooling
x = GlobalAveragePooling2D()(base_model.output)
# Mendefinisikan model yang mencakup semua lapisan sebelumnya dan Global Average Pooling
feature_extractor = Model(inputs=base_model.input, outputs=x)

# Load VGG16 model
base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x_vgg = GlobalAveragePooling2D()(base_model_vgg.output)
feature_extractor_vgg = Model(inputs=base_model_vgg.input, outputs=x_vgg)

# Load DenseNet201 model
base_model_densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x_densenet = GlobalAveragePooling2D()(base_model_densenet.output)
feature_extractor_densenet = Model(inputs=base_model_densenet.input, outputs=x_densenet)

# Fungsi untuk mengambil fitur dari model
def extract_features(img_path, model):
    # Membaca gambar
    img = image.load_img(img_path, target_size=(224, 224))
    # Mengubah gambar menjadi larik numpy
    img_array = image.img_to_array(img)
    # Menambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)
    # Pra-pemrosesan gambar sesuai dengan persyaratan MobileNetV2
    img_preprocessed = preprocess_input(img_array)
    # Mendapatkan fitur dari model
    features = model.predict(img_preprocessed)
    return features

# Memuat model SVM terbaik yang sudah disimpan sebelumnya
best_svm = joblib.load('best_svm_model_tuning.pkl')

# Kamus untuk mengonversi label prediksi menjadi nama lengkap kelas
class_names = {0: 'Melanoma', 1: 'BCC', 2: 'Normal'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Mendapatkan file gambar dari formulir
        img_file = request.files['image']
        # Menyimpan file gambar ke dalam direktori sementara
        img_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
        img_file.save(img_path)
        
        # Ekstrak fitur dari gambar menggunakan MobileNetV2
        image_features = extract_features(img_path, feature_extractor)
        # Memprediksi kelas gambar menggunakan model SVM
        prediction = best_svm.predict(image_features)
        # Mendapatkan nama lengkap kelas dari prediksi
        predicted_class = class_names[prediction[0]]
        
        # Membuat URL untuk gambar yang diunggah
        image_url = url_for('static', filename=f'uploads/{img_file.filename}')
        
        return render_template('index.html', predicted_class=predicted_class, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
