# history_service.py
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import base64
import io
from PIL import Image
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SugarCaneHistoryService:
    """
    Servicio para guardar y gestionar el historial de detecciones de caña de azúcar en MongoDB.
    Esta clase es completamente independiente y no afecta el funcionamiento de la API principal.
    """

    def __init__(self, connection_string: str = None, database_name: str = "sugar_cane_db"):
        """
        Inicializar el servicio de historial.

        Args:
            connection_string: String de conexión a MongoDB
            database_name: Nombre de la base de datos
        """
        self.connection_string = connection_string or os.getenv(
            'MONGO_CONNECTION_STRING',
            'mongodb://localhost:27017/'
        )
        self.database_name = database_name
        self.collection_name = "detection_history"

        self.client = None
        self.db = None
        self.collection = None

        self._connect()

    def _connect(self):
        """Establecer conexión con MongoDB"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Verificar conexión
            self.client.admin.command('ping')

            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]

            # Crear índices para optimizar consultas
            self._create_indexes()

            logger.info(f"Conectado exitosamente a MongoDB: {self.database_name}")

        except ConnectionFailure as e:
            logger.error(f"Error de conexión a MongoDB: {str(e)}")
            self.client = None
        except Exception as e:
            logger.error(f"Error inesperado al conectar: {str(e)}")
            self.client = None

    def _create_indexes(self):
        """Crear índices para optimizar las consultas"""
        try:
            # Índice por timestamp (para consultas cronológicas)
            self.collection.create_index("timestamp")

            # Índice por clase detectada
            self.collection.create_index("prediction.class_name")

            # Índice por nivel de confianza
            self.collection.create_index("prediction.confidence")

            # Índice compuesto para consultas complejas
            self.collection.create_index([
                ("timestamp", -1),
                ("prediction.class_name", 1)
            ])

            logger.info("Índices creados exitosamente")

        except Exception as e:
            logger.warning(f"Error al crear índices: {str(e)}")

    def is_connected(self) -> bool:
        """Verificar si la conexión está activa"""
        try:
            if self.client is None:
                return False
            self.client.admin.command('ping')
            return True
        except:
            return False

    def _image_to_base64(self, image_pil: Image.Image) -> str:
        """Convertir imagen PIL a base64 para almacenamiento"""
        try:
            # Redimensionar imagen para almacenamiento eficiente
            image_resized = image_pil.copy()
            image_resized.thumbnail((300, 300), Image.LANCZOS)

            # Convertir a base64
            img_buffer = io.BytesIO()
            image_resized.save(img_buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()

            return img_str
        except Exception as e:
            logger.error(f"Error al convertir imagen a base64: {str(e)}")
            return ""

    def save_detection(self,
                      image_pil: Image.Image,
                      prediction_result: Dict[str, Any],
                      processing_info: Dict[str, Any] = None,
                      user_info: Dict[str, Any] = None) -> Optional[str]:
        """
        Guardar una detección en el historial.

        Args:
            image_pil: Imagen PIL original
            prediction_result: Resultado de la predicción
            processing_info: Información adicional del procesamiento
            user_info: Información del usuario (opcional)

        Returns:
            ID del registro guardado o None si hay error
        """
        if not self.is_connected():
            logger.warning("No hay conexión a MongoDB. No se puede guardar la detección.")
            return None

        try:
            # Generar ID único
            detection_id = str(uuid.uuid4())

            # Preparar documento para MongoDB
            document = {
                "_id": detection_id,
                "timestamp": datetime.utcnow(),
                "prediction": {
                    "class_id": prediction_result.get("class_id"),
                    "class_name": prediction_result.get("class_name"),
                    "confidence": prediction_result.get("confidence"),
                    "all_predictions": prediction_result.get("all_predictions", {})
                },
                "image": {
                    "base64": self._image_to_base64(image_pil),
                    "original_size": image_pil.size,
                    "format": "JPEG"
                },
                "processing_info": processing_info or {},
                "user_info": user_info or {},
                "metadata": {
                    "api_version": "1.1.0",
                    "model_type": "DenseNet-161",
                    "created_at": datetime.utcnow().isoformat()
                }
            }

            # Guardar en MongoDB
            result = self.collection.insert_one(document)

            if result.acknowledged:
                logger.info(f"Detección guardada exitosamente: {detection_id}")
                return detection_id
            else:
                logger.error("Error al guardar la detección")
                return None

        except Exception as e:
            logger.error(f"Error al guardar detección: {str(e)}")
            return None

    def get_detection_by_id(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener una detección específica por ID.

        Args:
            detection_id: ID de la detección

        Returns:
            Documento de la detección o None
        """
        if not self.is_connected():
            return None

        try:
            document = self.collection.find_one({"_id": detection_id})
            if document:
                # Convertir ObjectId y datetime a strings para JSON
                document["timestamp"] = document["timestamp"].isoformat()
            return document
        except Exception as e:
            logger.error(f"Error al obtener detección {detection_id}: {str(e)}")
            return None

    def get_recent_detections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener las detecciones más recientes.

        Args:
            limit: Número máximo de detecciones a retornar

        Returns:
            Lista de detecciones
        """
        if not self.is_connected():
            return []

        try:
            cursor = self.collection.find().sort("timestamp", -1).limit(limit)
            detections = []

            for doc in cursor:
                doc["timestamp"] = doc["timestamp"].isoformat()
                # Remover imagen base64 para respuestas más ligeras
                if "image" in doc and "base64" in doc["image"]:
                    doc["image"]["has_image"] = True
                    del doc["image"]["base64"]
                detections.append(doc)

            return detections
        except Exception as e:
            logger.error(f"Error al obtener detecciones recientes: {str(e)}")
            return []

    def get_detections_by_class(self, class_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener detecciones filtradas por clase de enfermedad.

        Args:
            class_name: Nombre de la clase/enfermedad
            limit: Número máximo de resultados

        Returns:
            Lista de detecciones filtradas
        """
        if not self.is_connected():
            return []

        try:
            cursor = self.collection.find(
                {"prediction.class_name": class_name}
            ).sort("timestamp", -1).limit(limit)

            detections = []
            for doc in cursor:
                doc["timestamp"] = doc["timestamp"].isoformat()
                if "image" in doc and "base64" in doc["image"]:
                    doc["image"]["has_image"] = True
                    del doc["image"]["base64"]
                detections.append(doc)

            return detections
        except Exception as e:
            logger.error(f"Error al filtrar por clase {class_name}: {str(e)}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del historial de detecciones.

        Returns:
            Diccionario con estadísticas
        """
        if not self.is_connected():
            return {}

        try:
            # Contar total de detecciones
            total_detections = self.collection.count_documents({})

            # Estadísticas por clase
            pipeline = [
                {
                    "$group": {
                        "_id": "$prediction.class_name",
                        "count": {"$sum": 1},
                        "avg_confidence": {"$avg": "$prediction.confidence"}
                    }
                },
                {"$sort": {"count": -1}}
            ]

            class_stats = list(self.collection.aggregate(pipeline))

            # Detecciones por día (últimos 7 días)
            from datetime import timedelta
            seven_days_ago = datetime.utcnow() - timedelta(days=7)

            daily_pipeline = [
                {"$match": {"timestamp": {"$gte": seven_days_ago}}},
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": "$timestamp"
                            }
                        },
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"_id": 1}}
            ]

            daily_stats = list(self.collection.aggregate(daily_pipeline))

            return {
                "total_detections": total_detections,
                "detections_by_class": class_stats,
                "daily_detections_last_7_days": daily_stats,
                "last_updated": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error al obtener estadísticas: {str(e)}")
            return {}

    def delete_detection(self, detection_id: str) -> bool:
        """
        Eliminar una detección del historial.

        Args:
            detection_id: ID de la detección a eliminar

        Returns:
            True si se eliminó exitosamente
        """
        if not self.is_connected():
            return False

        try:
            result = self.collection.delete_one({"_id": detection_id})
            if result.deleted_count > 0:
                logger.info(f"Detección {detection_id} eliminada exitosamente")
                return True
            else:
                logger.warning(f"No se encontró la detección {detection_id}")
                return False
        except Exception as e:
            logger.error(f"Error al eliminar detección {detection_id}: {str(e)}")
            return False

    def close_connection(self):
        """Cerrar la conexión a MongoDB"""
        try:
            if self.client:
                self.client.close()
                logger.info("Conexión a MongoDB cerrada")
        except Exception as e:
            logger.error(f"Error al cerrar conexión: {str(e)}")

# Singleton para el servicio
_history_service_instance = None

def get_history_service() -> SugarCaneHistoryService:
    """
    Obtener instancia única del servicio de historial.

    Returns:
        Instancia del servicio de historial
    """
    global _history_service_instance

    if _history_service_instance is None:
        _history_service_instance = SugarCaneHistoryService()

    return _history_service_instance

# Función helper para facilitar el uso desde la API principal
def save_detection_to_history(image_pil: Image.Image,
                             prediction_result: Dict[str, Any],
                             processing_options: Dict[str, Any] = None,
                             user_info: Dict[str, Any] = None) -> Optional[str]:
    """
    Función helper para guardar detección desde la API principal.

    Esta función es thread-safe y maneja errores internamente.
    """
    try:
        service = get_history_service()
        return service.save_detection(
            image_pil=image_pil,
            prediction_result=prediction_result,
            processing_info=processing_options,
            user_info=user_info
        )
    except Exception as e:
        logger.error(f"Error en save_detection_to_history: {str(e)}")
        return None