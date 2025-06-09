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
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torchvision.models import densenet161
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from contextlib import asynccontextmanager
import cv2
from fastapi.responses import StreamingResponse
import json
from datetime import datetime

# Cargar variables de entorno
from dotenv import load_dotenv

load_dotenv()

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
class UserInfo(BaseModel):
    id: Optional[int] = None
    email: Optional[str] = None
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    middleName: Optional[str] = None
    phoneNumber: Optional[str] = None
    roles: Optional[List[Dict[str, Any]]] = None
    isActive: Optional[bool] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    profileImage: Optional[str] = None


class PredictionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    all_predictions: Dict[str, float]
    history_id: Optional[str] = None
    user_authenticated: bool = False
    user_info: Optional[Dict[str, Any]] = None


class PredictionResultWithHistory(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    all_predictions: Dict[str, float]
    history_id: Optional[str] = None


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


def extract_user_from_token(authorization: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Extrae información del usuario desde el header Authorization o token
    """
    if not authorization:
        return None

    try:
        # Si es un Bearer token, extraer solo el token
        if authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "")
        else:
            token = authorization

        # Aquí puedes decodificar el JWT si necesitas validarlo
        # Por ahora, asumimos que el frontend envía la info del usuario directamente
        return {"token": token, "token_provided": True}
    except Exception as e:
        print(f"Error al procesar token: {str(e)}")
        return None


def get_prediction_with_history(image, use_advanced_preprocessing=True, processing_options=None, user_info=None):
    """
    Función de predicción mejorada que guarda automáticamente en el historial.

    Args:
        image: Imagen PIL
        use_advanced_preprocessing: Si usar preprocesamiento avanzado
        processing_options: Opciones de procesamiento
        user_info: Información del usuario (opcional)

    Returns:
        Resultado de la predicción con ID del historial
    """
    try:
        # --- PREPROCESAMIENTO ---
        # Convertir PIL a OpenCV (BGR)
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Aplicar mejora de contraste si está habilitado
        if use_advanced_preprocessing and (
                processing_options is None or processing_options.get("enhance_contrast", True)):
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

        # Preparar resultado
        result = {
            "class_id": pred_class,
            "class_name": DISEASE_MAPPING[pred_class],
            "confidence": confidence,
            "all_predictions": all_probs
        }

        # Guardar en historial (asíncrono, no bloquea la respuesta)
        try:
            # Importar aquí para evitar errores si MongoDB no está disponible
            from history_service import save_detection_to_history

            history_id = save_detection_to_history(
                image_pil=image,
                prediction_result=result,
                processing_options=processing_options,
                user_info=user_info
            )

            if history_id:
                result["history_id"] = history_id
                if user_info and user_info.get("is_authenticated"):
                    user_email = user_info.get("user", {}).get("email", "N/A")
                    print(f"Detección guardada en historial con ID: {history_id} para usuario: {user_email}")
                else:
                    print(f"Detección guardada en historial con ID: {history_id} (usuario anónimo)")
            else:
                print("Advertencia: No se pudo guardar en el historial")

        except Exception as history_error:
            # No fallar la predicción si hay error en el historial
            print(f"Error al guardar en historial: {str(history_error)}")

        return result

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
    description="API para clasificación de enfermedades de la caña de azúcar usando DenseNet-161 con preprocesamiento avanzado y historial de usuarios",
    version="1.2.0",
    lifespan=lifespan
)

# Configuración CORS actualizada
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aivision-web.netlify.app",
        "http://localhost:3000",  # Para desarrollo local
        "http://localhost:5173",  # Para desarrollo local con Vite
        "http://localhost:8080",  # spring boot
        "http://localhost:4200"  # de angular
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Variable global para el modelo
model = None

# Incluir router de historial DESPUÉS de crear la app
try:
    from history_endpoints import history_router

    app.include_router(history_router)
    print("✅ Router de historial incluido exitosamente")
except ImportError as e:
    print(f"⚠️ Advertencia: No se pudo cargar el router de historial: {str(e)}")
    print("La API funcionará sin funcionalidades de historial")


# ============= ENDPOINTS DE LA API =============
@app.get("/")
def home_with_history():
    return {
        "mensaje": "API mejorada de clasificación de enfermedades de la caña de azúcar con historial de usuarios",
        "modelo": "DenseNet-161",
        "preprocesamiento": "Avanzado (mejora de contraste + eliminación de fondo)",
        "historial": "MongoDB integrado para guardar todas las detecciones con información de usuario",
        "enfermedades_detectables": DISEASE_MAPPING,
        "caracteristicas": {
            "historial_usuarios": "Guarda información completa del usuario autenticado",
            "usuarios_anonimos": "Admite detecciones sin autenticación",
            "metadatos_completos": "Guarda ubicación, dispositivo, timestamp, etc."
        },
        "endpoints": [
            {"ruta": "/predict", "método": "POST", "descripción": "Predicción simple con información de usuario"},
            {"ruta": "/advanced-predict", "método": "POST",
             "descripción": "Predicción avanzada con opciones personalizables"},
            {"ruta": "/preview", "método": "POST", "descripción": "Previsualizar etapas de procesamiento"},
            {"ruta": "/history/recent", "método": "GET", "descripción": "Obtener detecciones recientes"},
            {"ruta": "/history/statistics", "método": "GET", "descripción": "Estadísticas del historial"},
            {"ruta": "/history/user/{user_id}", "método": "GET", "descripción": "Historial por usuario específico"},
            {"ruta": "/history/user/{user_id}/statistics", "método": "GET", "descripción": "Estadísticas por usuario"},
            {"ruta": "/history/authenticated", "método": "GET", "descripción": "Solo detecciones autenticadas"},
            {"ruta": "/history/by-class/{class_name}", "método": "GET",
             "descripción": "Filtrar por tipo de enfermedad"},
            {"ruta": "/history/{detection_id}", "método": "GET", "descripción": "Obtener detección específica"},
            {"ruta": "/history/{detection_id}/image", "método": "GET", "descripción": "Obtener imagen de detección"},
            {"ruta": "/history/health/check", "método": "GET", "descripción": "Estado del servicio de historial"}
        ]
    }


@app.post("/predict", response_model=PredictionResult)
async def predict_with_user_info(
        file: UploadFile = File(...),
        user_info: Optional[str] = Query(None, description="Información del usuario como JSON string"),
        authorization: Optional[str] = Header(None, description="Token de autorización"),
        location: Optional[str] = Query(None, description="Ubicación de la detección"),
        device_info: Optional[str] = Query(None, description="Información del dispositivo")
):
    """
    Endpoint mejorado que guarda información del usuario en el historial.

    Parámetros:
    - file: Imagen a analizar
    - user_info: JSON string con información del usuario desde localStorage
    - authorization: Token de autorización (opcional)
    - location: Ubicación donde se tomó la imagen (opcional)
    - device_info: Información del dispositivo (opcional)
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

        # Procesar información del usuario
        user_data = None
        is_authenticated = False

        if user_info:
            try:
                user_data = json.loads(user_info)
                is_authenticated = True
                print(f"Usuario autenticado: {user_data.get('email', 'N/A')} - ID: {user_data.get('id', 'N/A')}")
            except json.JSONDecodeError:
                print("Error al decodificar información del usuario")

        # Extraer información del token si está disponible
        token_info = extract_user_from_token(authorization)

        # Configurar opciones de procesamiento
        processing_options = {
            "enhance_contrast": True,
            "advanced_preprocessing": True,
            "image_size": image.size,
            "file_size": len(content),
            "processing_timestamp": datetime.utcnow().isoformat()
        }

        # Información completa para el historial
        complete_user_info = {
            "source": "predict_endpoint",
            "filename": file.filename,
            "content_type": file.content_type,
            "location": location,
            "device_info": device_info,
            "is_authenticated": is_authenticated,
            "authentication": {
                "has_token": token_info is not None,
                "token_info": token_info
            }
        }

        # Agregar información del usuario si está disponible
        if user_data:
            complete_user_info["user"] = {
                "id": user_data.get("id"),
                "email": user_data.get("email"),
                "firstName": user_data.get("firstName"),
                "lastName": user_data.get("lastName"),
                "middleName": user_data.get("middleName"),
                "fullName": f"{user_data.get('firstName', '')} {user_data.get('middleName', '')} {user_data.get('lastName', '')}".strip(),
                "phoneNumber": user_data.get("phoneNumber"),
                "isActive": user_data.get("isActive"),
                "createdAt": user_data.get("createdAt"),
                "updatedAt": user_data.get("updatedAt"),
                "profileImage": user_data.get("profileImage"),
                "roles": user_data.get("roles", []),
                "role_names": [role.get("name") for role in user_data.get("roles", []) if role.get("name")]
            }

        # Realizar predicción y guardar en historial
        result = get_prediction_with_history(
            image=image,
            use_advanced_preprocessing=True,
            processing_options=processing_options,
            user_info=complete_user_info
        )

        # Preparar respuesta con información del usuario
        response = PredictionResult(
            class_id=result["class_id"],
            class_name=result["class_name"],
            confidence=result["confidence"],
            all_predictions=result["all_predictions"],
            history_id=result.get("history_id"),
            user_authenticated=is_authenticated,
            user_info=user_data if is_authenticated else None
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")


@app.post("/advanced-predict", response_model=PredictionResult)
async def advanced_predict_with_user_info(
        file: UploadFile = File(...),
        enhance_contrast: bool = Query(True, description="Aplicar mejora de contraste adaptativo"),
        user_info: Optional[str] = Query(None, description="Información del usuario como JSON string"),
        authorization: Optional[str] = Header(None, description="Token de autorización"),
        location: Optional[str] = Query(None, description="Ubicación de la detección"),
        device_info: Optional[str] = Query(None, description="Información del dispositivo"),
        notes: Optional[str] = Query(None, description="Notas adicionales sobre la detección"),
        plant_id: Optional[str] = Query(None, description="ID de la planta analizada"),
        field_section: Optional[str] = Query(None, description="Sección del campo")
):
    """
    Endpoint avanzado que guarda información completa del usuario y contexto.

    Parámetros adicionales:
    - enhance_contrast: Si aplicar mejora de contraste
    - user_info: JSON string con información completa del usuario
    - authorization: Token JWT del usuario
    - location: Ubicación GPS o descripción del lugar
    - device_info: Información del dispositivo usado
    - notes: Notas adicionales sobre la detección
    - plant_id: ID de la planta específica (si aplica)
    - field_section: Sección específica del campo
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

        # Procesar información del usuario
        user_data = None
        is_authenticated = False

        if user_info:
            try:
                user_data = json.loads(user_info)
                is_authenticated = True
                print(f"Usuario autenticado: {user_data.get('email', 'N/A')} - ID: {user_data.get('id', 'N/A')}")
                print(f"Roles: {[role.get('name') for role in user_data.get('roles', [])]}")
            except json.JSONDecodeError:
                print("Error al decodificar información del usuario")

        # Extraer información del token
        token_info = extract_user_from_token(authorization)

        # Configurar opciones de procesamiento
        processing_options = {
            "enhance_contrast": enhance_contrast,
            "advanced_preprocessing": True,
            "image_size": image.size,
            "file_size": len(content),
            "processing_timestamp": datetime.utcnow().isoformat(),
            "notes": notes,
            "plant_id": plant_id,
            "field_section": field_section
        }

        # Información completa para el historial
        complete_user_info = {
            "source": "advanced_predict_endpoint",
            "filename": file.filename,
            "content_type": file.content_type,
            "location": location,
            "device_info": device_info,
            "notes": notes,
            "plant_id": plant_id,
            "field_section": field_section,
            "is_authenticated": is_authenticated,
            "authentication": {
                "has_token": token_info is not None,
                "token_info": token_info
            }
        }

        # Agregar información detallada del usuario
        if user_data:
            complete_user_info["user"] = {
                "id": user_data.get("id"),
                "email": user_data.get("email"),
                "firstName": user_data.get("firstName"),
                "lastName": user_data.get("lastName"),
                "middleName": user_data.get("middleName"),
                "fullName": f"{user_data.get('firstName', '')} {user_data.get('middleName', '')} {user_data.get('lastName', '')}".strip(),
                "phoneNumber": user_data.get("phoneNumber"),
                "isActive": user_data.get("isActive"),
                "createdAt": user_data.get("createdAt"),
                "updatedAt": user_data.get("updatedAt"),
                "profileImage": user_data.get("profileImage"),
                "roles": user_data.get("roles", []),
                "role_names": [role.get("name") for role in user_data.get("roles", []) if role.get("name")],
                "plants": user_data.get("plants", []),
                "detections": user_data.get("detections", [])
            }

        # Realizar predicción y guardar en historial
        result = get_prediction_with_history(
            image=image,
            use_advanced_preprocessing=True,
            processing_options=processing_options,
            user_info=complete_user_info
        )

        # Preparar respuesta
        response = PredictionResult(
            class_id=result["class_id"],
            class_name=result["class_name"],
            confidence=result["confidence"],
            all_predictions=result["all_predictions"],
            history_id=result.get("history_id"),
            user_authenticated=is_authenticated,
            user_info=user_data if is_authenticated else None
        )

        return response

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
def test_api(image_path, api_url="http://localhost:8001", endpoint="/advanced-predict"):
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

        # Simular información de usuario para prueba
        test_user = {
            "id": 1,
            "email": "test@example.com",
            "firstName": "Usuario",
            "lastName": "Prueba",
            "middleName": "De",
            "roles": [{"id": 1, "name": "USER"}]
        }

        # Parámetros de la solicitud
        params = {
            "user_info": json.dumps(test_user),
            "location": "Campo de prueba",
            "device_info": "Test Device",
            "notes": "Prueba desde script"
        }

        if endpoint == "/advanced-predict":
            params.update({
                "enhance_contrast": "true"
            })

        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(f"{api_url}{endpoint}", files=files, params=params)

        elapsed = time.time() - start_time

        if response.status_code != 200:
            print(f"Error en la predicción: {response.json()}")
            return

        result = response.json()
        print("\n" + "=" * 50)
        print("RESULTADOS DE LA PREDICCIÓN")
        print("=" * 50)
        print(f"Diagnóstico: {result['class_name']}")
        print(f"Confianza: {result['confidence'] * 100:.2f}%")
        print(f"Tiempo de respuesta: {elapsed:.3f} segundos")

        if 'history_id' in result:
            print(f"ID en historial: {result['history_id']}")

        if result.get('user_authenticated'):
            print(f"Usuario autenticado: {result.get('user_info', {}).get('email', 'N/A')}")
        else:
            print("Detección realizada como usuario anónimo")

        print("\nProbabilidades por clase:")

        sorted_probs = sorted(result['all_predictions'].items(),
                              key=lambda x: x[1],
                              reverse=True)

        for disease, prob in sorted_probs:
            print(f"- {disease}: {prob * 100:.2f}%")

        print("=" * 50)

        # Mostrar información adicional si está disponible
        if result.get('user_authenticated') and result.get('user_info'):
            user_info = result['user_info']
            print(f"\nInformación del usuario:")
            print(f"- Nombre: {user_info.get('firstName', '')} {user_info.get('lastName', '')}")
            print(f"- Email: {user_info.get('email', '')}")
            print(f"- Roles: {[role.get('name') for role in user_info.get('roles', [])]}")

    except Exception as e:
        print(f"Error al realizar la predicción: {str(e)}")


# ============= PUNTO DE ENTRADA PRINCIPAL =============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="API de clasificación de enfermedades de caña de azúcar con información de usuario")
    parser.add_argument("--image_path", help="Ruta a la imagen para analizar")
    parser.add_argument("--url", default="http://localhost:8001", help="URL base de la API")
    parser.add_argument("--endpoint", default="/advanced-predict",
                        choices=["/predict", "/advanced-predict"],
                        help="Endpoint para la predicción")

    args = parser.parse_args()

    # Si se proporciona una imagen, ejecutar el cliente de prueba
    if args.image_path:
        test_api(args.image_path, args.url, args.endpoint)
    else:
        # Si no hay imagen, iniciar el servidor API
        print("🚀 Iniciando API de detección de enfermedades de caña de azúcar...")
        print("📊 Características:")
        print("   - Historial de detecciones en MongoDB")
        print("   - Información completa de usuarios autenticados")
        print("   - Soporte para usuarios anónimos")
        print("   - Metadatos completos (ubicación, dispositivo, notas)")
        print("   - Endpoints especializados por usuario")
        print(f"🌐 Servidor disponible en: http://127.0.0.1:8001")
        print(f"📚 Documentación en: http://127.0.0.1:8001/docs")
        print(f"🗃️ MongoDB Express en: http://localhost:8081")

        port = int(os.getenv("PORT", 8001))
        uvicorn.run(app, host="127.0.0.1", port=port)