import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from PIL import Image

# Configuración de directorios
DATA_DIR = r'C:\Users\front.dos\Documents\Proyectos\layer\data'
MODEL_DIR = 'models'
MODEL_NAME = 'face_recognition_model.pkl'

# Preprocesar las imágenes y extraer características
def preprocess_images(data_dir):
    images = []
    labels = []

    # Recorremos las subcarpetas en 'data/train'
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):  # Solo entramos si es una carpeta
            # Recorremos las imágenes dentro de cada subcarpeta
            for image_name in os.listdir(label_dir):
                if image_name.endswith('.jpg') or image_name.endswith('.png'):
                    image_path = os.path.join(label_dir, image_name)
                    img = Image.open(image_path)
                    img = img.convert('L')  # Convertir a escala de grises
                    img = img.resize((100, 100))  # Redimensionar a un tamaño estándar
                    img_array = np.array(img).flatten()  # Aplanar la imagen
                    images.append(img_array)
                    labels.append(label)

    return np.array(images), np.array(labels)

# Entrenar el modelo
def train_model():
    # Cargar y preprocesar las imágenes desde 'data/train'
    images, labels = preprocess_images(os.path.join(DATA_DIR, 'train'))

    # Codificar las etiquetas (nombres) en números
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Crear el clasificador SVM
    clf = SVC(kernel='linear', probability=True)

    # Entrenar el modelo
    clf.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Guardar el modelo y el codificador de etiquetas
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, MODEL_NAME))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    print(f"Modelo guardado en {os.path.join(MODEL_DIR, MODEL_NAME)}")

# Cargar el modelo y el codificador de etiquetas
def load_model():
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')

    if os.path.exists(model_path):
        clf = joblib.load(model_path)
        label_encoder = joblib.load(label_encoder_path)
        print("Modelo y codificador cargados.")
    else:
        print("Modelo no encontrado, entrenando...")
        train_model()
        clf = joblib.load(model_path)
        label_encoder = joblib.load(label_encoder_path)

    return clf, label_encoder

# Función para predecir la clase de una imagen
def predict_image(image_path, clf, label_encoder):
    img = Image.open(image_path)
    img = img.convert('L')  # Convertir a escala de grises
    img = img.resize((100, 100))  # Redimensionar
    img_array = np.array(img).flatten()  # Aplanar la imagen

    # Realizar la predicción
    prediction = clf.predict([img_array])
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    return predicted_class

# Función principal
if __name__ == "__main__":
    # Cargar el modelo y el codificador
    clf, label_encoder = load_model()

    # Hacer una predicción sobre una nueva imagen
    image_path = 'data/test/test_image.jpg'  # Cambia a la ruta de tu imagen de prueba
    predicted_class = predict_image(image_path, clf, label_encoder)
    print(f"Predicción para la imagen '{image_path}': {predicted_class}")
