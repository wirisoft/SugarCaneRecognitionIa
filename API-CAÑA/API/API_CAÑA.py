import os
import io
import torch
import uvicorn
import numpy as np
import requests
import argparse
import time
from PIL import Image
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torchvision.models import densenet161
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from contextlib import asynccontextmanager
import cv2
from fastapi.responses import StreamingResponse

# ============= CONFIGURACIÓN DEL MODELO =============
MODEL_PATH = os.getenv('MODEL_PATH', r"C:\Users\misju\SugarCaneRecognitionIa\rpeca\API-CAÑA\API\model_epoch_18.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapeo de índices a nombres de enfermedades de la caña
DISEASE_MAPPING = {
    0: 'healthy_cane (HC)',
    1: 'leaf_scald (LS)',
    2: 'mosaic_virus (MV)',
    3: 'painted_fly (PF)',
    4: 'red_rot (RR)',
    5: 'roya (R)',
    6: 'yellow (Y)'
}

# Transformaciones para preprocesamiento de imágenes
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============= DEFINICIÓN DEL MODELO =============
class SugarCaneDenseNetModel(nn.Module):
    def __init__(self):
        super(SugarCaneDenseNetModel, self).__init__()
        # Use densenet161 to match the training configuration
        self.densenet = densenet161(pretrained=False)
        # Modify classifier for 7 classes
        self.densenet.classifier = nn.Sequential(
            nn.Linear(2208, 1024),  # DenseNet-161 has 2208 features
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 7)  # 7 classes for different sugar cane diseases
        )
    
    def forward(self, x):
        return self.densenet(x)

# ============= MODELOS DE RESPUESTA =============
class PredictionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    all_predictions: Dict[str, float]

class ErrorResponse(BaseModel):
    detail: str

class ProcessingOptions(BaseModel):
    enhance_contrast: bool = True

# ============= FUNCIONES DE PREPROCESAMIENTO AVANZADO =============
def preserve_color_enhancement(image):
    """
    Mejorar el contraste preservando los colores originales de la hoja de caña.
    Recibe una imagen en formato BGR (OpenCV).
    """
    # Convertir a LAB para separar la luminosidad del color
    lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplicar CLAHE solo al canal de luminancia con parámetros optimizados para hojas
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Fusionar de nuevo con los canales de color originales
    enhanced_lab = cv2.merge([cl, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr

# ============= FUNCIONES AUXILIARES =============
def load_model():
    global model
    if model is None:
        print(f"Cargando modelo desde {MODEL_PATH}...")
        model = SugarCaneDenseNetModel()
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model = model.to(DEVICE)
            model.eval()
            print(f"Modelo cargado exitosamente. Usando dispositivo: {DEVICE}")
        except Exception as e:
            model = None
            print(f"Error al cargar el modelo: {str(e)}")
            raise RuntimeError(f"No se pudo cargar el modelo: {str(e)}")

def get_prediction(image, use_advanced_preprocessing=True, processing_options=None):
    try:
        # --- PREPROCESAMIENTO ---
        # Convertir PIL a OpenCV (BGR)
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        
        # Aplicar mejora de contraste si está habilitado
        if use_advanced_preprocessing and (processing_options is None or processing_options.get("enhance_contrast", True)):
            image_cv = preserve_color_enhancement(image_cv)
        
        # Convertir de vuelta a RGB para PIL
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        processed_image = Image.fromarray(image_rgb)
        
        # Preprocesar la imagen para el modelo
        img_tensor = image_transforms(processed_image).unsqueeze(0).to(DEVICE)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        pred_class = torch.argmax(probabilities).item()
        confidence = probabilities[pred_class].item()
        all_probs = {DISEASE_MAPPING[i]: prob.item() for i, prob in enumerate(probabilities)}
        
        return {
            "class_id": pred_class,
            "class_name": DISEASE_MAPPING[pred_class],
            "confidence": confidence,
            "all_predictions": all_probs
        }
    except Exception as e:
        print(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en procesamiento de imagen: {str(e)}")

def image_to_bytes(image_pil):
    """Convierte una imagen PIL a bytes para streaming"""
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def process_stages(image, use_advanced=True, processing_options=None):
    """
    Procesa la imagen en diferentes etapas y retorna un diccionario con los resultados
    
    Args:
        image: Imagen PIL en formato RGB
        use_advanced: Si se debe usar el preprocesamiento avanzado
        processing_options: Opciones específicas para el preprocesamiento
    
    Returns:
        Diccionario con las diferentes etapas del procesamiento
    """
    results = {}
    
    # Etapa 1: Imagen original redimensionada
    resized = image.resize((512, 512), Image.LANCZOS)
    results["original"] = resized
    
    if use_advanced:
        # Convertir PIL a OpenCV (BGR)
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        
        # Etapa 2: Solo mejora de contraste
        contrast_enhanced = preserve_color_enhancement(image_cv)
        contrast_enhanced_rgb = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2RGB)
        results["enhanced"] = Image.fromarray(contrast_enhanced_rgb)
        
        # Etapa 3: Imagen final para el modelo (reescalada y normalizada)
        model_input = image_transforms(results["enhanced"])
        model_preview = transforms.ToPILImage()(model_input)
        results["model_input"] = model_preview
    else:
        # Preprocesamiento para el modelo
        preprocessed = image_transforms(resized)
        preprocessed_img = transforms.ToPILImage()(preprocessed)
        results["preprocessed"] = preprocessed_img
    
    return results

# ============= CONFIGURACIÓN DE LA API =============
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        load_model()
    except Exception as e:
        print(f"Advertencia en inicio: {str(e)}")
    yield
    # Shutdown
    # Aquí puedes agregar código de limpieza si es necesario

app = FastAPI(
    title="Sugar Cane Disease Classification API",
    description="API para clasificación de enfermedades de la caña de azúcar usando ViT-Large-32 con preprocesamiento avanzado",
    version="1.1.0",
    lifespan=lifespan
)

# Configuración CORS actualizada
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aivision-web.netlify.app",
        "http://localhost:3000",  # Para desarrollo local
        "http://localhost:5173"   # Para desarrollo local con Vite
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Variable global para el modelo
model = None

# ============= ENDPOINTS DE LA API =============
@app.get("/")
def home():
    return {
        "mensaje": "API mejorada de clasificación de enfermedades de la caña de azúcar",
        "modelo": "ViT-Large-32",
        "preprocesamiento": "Avanzado (mejora de contraste + eliminación de fondo)",
        "enfermedades_detectables": DISEASE_MAPPING,
        "endpoints": [
            {"ruta": "/predict", "método": "POST", "descripción": "Realizar una predicción con una imagen de hoja de caña"},
            {"ruta": "/advanced-predict", "método": "POST", "descripción": "Predicción con opciones de preprocesamiento personalizables"},
            {"ruta": "/preview", "método": "POST", "descripción": "Previsualizar etapas de procesamiento de imagen"},
        ]
    }

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    """
    Endpoint original para mantener compatibilidad.
    Usa el nuevo preprocesamiento avanzado por defecto.
    """
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {str(e)}")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen (jpg, jpeg, png)")
    
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        result = get_prediction(image, use_advanced_preprocessing=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")

@app.post("/advanced-predict", response_model=PredictionResult)
async def advanced_predict(
    file: UploadFile = File(...),
    enhance_contrast: bool = Query(True, description="Aplicar mejora de contraste adaptativo")
):
    """
    Endpoint para predicción con opciones de preprocesamiento personalizables.
    Permite configurar el proceso de mejora de imagen.
    """
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {str(e)}")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen (jpg, jpeg, png)")
    
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Configurar opciones de procesamiento
        processing_options = {
            "enhance_contrast": enhance_contrast
        }
        
        result = get_prediction(image, use_advanced_preprocessing=True, processing_options=processing_options)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")

@app.post("/preview")
async def preview_processing(
    file: UploadFile = File(...),
    stage: str = Query("enhanced", description="Etapa de procesamiento a visualizar"),
    use_advanced: bool = Query(True, description="Usar preprocesamiento avanzado")
):
    """
    Previsualizar cada etapa del procesamiento de imágenes.
    
    Etapas disponibles con preprocesamiento avanzado:
    - original: Imagen original redimensionada
    - enhanced: Imagen con mejora de contraste
    - model_input: Imagen final que ve el modelo (224x224 normalizada)
    
    Etapas con preprocesamiento básico:
    - original: Imagen original redimensionada
    - preprocessed: Imagen final para el modelo
    """
    try:
        # Verificar formato de imagen
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser una imagen (jpg, jpeg, png)"
            )
        
        # Leer y procesar la imagen
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Procesar la imagen en diferentes etapas
        processed_images = process_stages(
            image, 
            use_advanced=use_advanced,
            processing_options=None  # Opciones predeterminadas
        )
        
        # Verificar que la etapa solicitada existe
        if stage not in processed_images:
            valid_stages = list(processed_images.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Etapa '{stage}' no válida. Opciones: {', '.join(valid_stages)}"
            )
        
        # Convertir la imagen procesada a bytes y enviarla
        img_bytes = image_to_bytes(processed_images[stage])
        
        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="processed_{stage}.png"'
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en procesamiento de imagen: {str(e)}"
        )

@app.get("/healthcheck")
def health_check():
    if model is None:
        try:
            load_model()
            return {"status": "ok", "modelo": "cargado", "device": str(DEVICE)}
        except:
            return {"status": "error", "mensaje": "Modelo no disponible", "device": str(DEVICE)}
    else:
        return {"status": "ok", "modelo": "cargado", "device": str(DEVICE)}

# ============= CÓDIGO DE PRUEBA =============
def test_api(image_path, api_url="http://localhost:8000", endpoint="/advanced-predict"):
    if not os.path.exists(image_path):
        print(f"Error: La imagen no existe en la ruta: {image_path}")
        return
    
    try:
        img = Image.open(image_path)
        img.close()
    except:
        print(f"Error: El archivo no es una imagen válida: {image_path}")
        return
    
    try:
        health_response = requests.get(f"{api_url}/healthcheck")
        health_data = health_response.json()
        
        if health_response.status_code != 200 or health_data.get("status") != "ok":
            print(f"Error: La API no está disponible o el modelo no está cargado.")
            print(f"Respuesta: {health_data}")
            return
        
        print(f"API funcionando correctamente. Dispositivo: {health_data.get('device')}")
    except Exception as e:
        print(f"Error al conectar con la API: {str(e)}")
        return
    
    try:
        print(f"Enviando imagen {os.path.basename(image_path)} para predicción...")
        start_time = time.time()
        
        # Parámetros de la solicitud
        params = {}
        if endpoint == "/advanced-predict":
            params = {
                "enhance_contrast": "true"
            }
        
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(f"{api_url}{endpoint}", files=files, params=params)
        
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            print(f"Error en la predicción: {response.json()}")
            return
        
        result = response.json()
        print("\n" + "="*50)
        print("RESULTADOS DE LA PREDICCIÓN")
        print("="*50)
        print(f"Diagnóstico: {result['class_name']}")
        print(f"Confianza: {result['confidence']*100:.2f}%")
        print(f"Tiempo de respuesta: {elapsed:.3f} segundos")
        print("\nProbabilidades por clase:")
        
        sorted_probs = sorted(result['all_predictions'].items(), 
                            key=lambda x: x[1], 
                            reverse=True)
        
        for disease, prob in sorted_probs:
            print(f"- {disease}: {prob*100:.2f}%")
        
        print("="*50)
        
        # Previsualizar las etapas de procesamiento
        print("\nGenerando previsualizaciones de procesamiento...")
        stages = ["original", "enhanced", "preprocessed"]
        
        for stage in stages:
            try:
                preview_response = requests.post(
                    f"{api_url}/preview",
                    files=files,
                    params={"stage": stage, "use_advanced": "true"}
                )
                
                if preview_response.status_code == 200:
                    output_path = f"preview_{stage}_{os.path.basename(image_path)}"
                    with open(output_path, "wb") as out:
                        out.write(preview_response.content)
                    print(f"  - Previsualización '{stage}' guardada como: {output_path}")
            except Exception as e:
                print(f"  - Error al generar previsualización '{stage}': {str(e)}")
        
    except Exception as e:
        print(f"Error al realizar la predicción: {str(e)}")

# ============= PUNTO DE ENTRADA PRINCIPAL =============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cliente de prueba para la API de clasificación ocular")
    parser.add_argument("--image_path", help="Ruta a la imagen para analizar")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base de la API")


    parser.add_argument("--endpoint", default="/advanced-predict", 
                        choices=["/predict", "/advanced-predict"],
                        help="Endpoint para la predicción")
    
    args = parser.parse_args()
    
    # Si se proporciona una imagen, ejecutar el cliente de prueba
    if args.image_path:
        test_api(args.image_path, args.url, args.endpoint)
    else:
        # Si no hay imagen, iniciar el servidor API
        port = int(os.getenv("PORT", 8001))
        uvicorn.run(app, host="127.0.0.1", port=port)