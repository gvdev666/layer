import cv2  # Biblioteca para detección de rostros
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import joblib
import os
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configuración
DATA_DIR = 'data'
MODEL_DIR = 'models'
MODEL_NAME = 'face_recognition_model.pkl'
LABEL_ENCODER_NAME = 'label_encoder.pkl'
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Crear instancia de la aplicación
app = FastAPI()

# Cargar modelo y codificador
def load_model():
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    label_encoder_path = os.path.join(MODEL_DIR, LABEL_ENCODER_NAME)

    if not os.path.exists(model_path) or not os.path.exists(label_encoder_path):
        raise FileNotFoundError("El modelo o el codificador no están entrenados. Entrena primero el modelo.")

    clf = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    return clf, label_encoder

# Preprocesar imagen
def preprocess_image(image):
    img = Image.open(image)
    img = img.convert('L')  # Convertir a escala de grises
    img = img.resize((100, 100))  # Redimensionar
    img_array = np.array(img).flatten()  # Aplanar la imagen
    return img_array

# Detección de rostro
def detect_faces(image):
    # Convertir la imagen a formato OpenCV
    img_array = np.array(Image.open(image).convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Cargar el clasificador de rostros
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Devolver las coordenadas de los rostros detectados
    return faces

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Validar formato del archivo
    if not (file.filename.endswith(".jpg") or file.filename.endswith(".png")):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Usa .jpg o .png")

    # Detectar rostros
    faces = detect_faces(file.file)
    if len(faces) == 0:
        return {"message": "No se detectaron rostros en la imagen"}

    # Cargar modelo y codificador
    try:
        clf, label_encoder = load_model()
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    # Procesar imagen (usamos el primer rostro detectado)
    try:
        face_array = preprocess_image(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando la imagen: {e}")

    # Hacer predicción
    prediction = clf.predict([face_array])
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    return {"message": "Rostro detectado", "prediction": predicted_class}

@app.post("/train/")
async def train_model():
    # Preprocesar datos
    images, labels = [], []
    train_dir = os.path.join(DATA_DIR, 'train')

    if not os.path.exists(train_dir):
        raise HTTPException(status_code=400, detail="Directorio de entrenamiento no encontrado")

    for label in os.listdir(train_dir):
        label_dir = os.path.join(train_dir, label)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                if image_name.endswith(".jpg") or image_name.endswith(".png"):
                    image_path = os.path.join(label_dir, image_name)
                    img = preprocess_image(image_path)
                    images.append(img)
                    labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # Entrenar modelo
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train, y_train)

    # Evaluar modelo
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    # Guardar modelo y codificador
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, MODEL_NAME))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, LABEL_ENCODER_NAME))
    return {"message": "Modelo entrenado y guardado correctamente", "report": report}

# Ruta para verificar la API
@app.get("/")
def home():
    return {"message": "API de reconocimiento facial con detección de rostros"}
