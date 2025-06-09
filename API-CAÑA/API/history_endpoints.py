# history_endpoints.py
from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import base64
import io
from PIL import Image
from fastapi.responses import StreamingResponse

from history_service import get_history_service, SugarCaneHistoryService


class DetectionHistoryResponse(BaseModel):
    id: str
    timestamp: str
    prediction: Dict[str, Any]
    processing_info: Dict[str, Any]
    user_info: Dict[str, Any]
    metadata: Dict[str, Any]
    has_image: bool


class StatisticsResponse(BaseModel):
    total_detections: int
    detections_by_class: List[Dict[str, Any]]
    daily_detections_last_7_days: List[Dict[str, Any]]
    last_updated: str


class DeleteResponse(BaseModel):
    success: bool
    message: str


# Crear router para endpoints de historial
history_router = APIRouter(prefix="/history", tags=["historial"])


@history_router.get("/recent", response_model=List[DetectionHistoryResponse])
async def get_recent_detections(
        limit: int = Query(10, ge=1, le=100, description="Número de detecciones a retornar")
):
    """
    Obtener las detecciones más recientes del historial.
    """
    try:
        service = get_history_service()

        if not service.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Servicio de historial no disponible. Verifique la conexión a MongoDB."
            )

        detections = service.get_recent_detections(limit=limit)

        # Convertir a modelo de respuesta
        response_data = []
        for detection in detections:
            response_data.append(DetectionHistoryResponse(
                id=detection["_id"],
                timestamp=detection["timestamp"],
                prediction=detection["prediction"],
                processing_info=detection.get("processing_info", {}),
                user_info=detection.get("user_info", {}),
                metadata=detection.get("metadata", {}),
                has_image=detection.get("image", {}).get("has_image", False)
            ))

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener historial: {str(e)}")


@history_router.get("/by-class/{class_name}", response_model=List[DetectionHistoryResponse])
async def get_detections_by_class(
        class_name: str = Path(..., description="Nombre de la clase de enfermedad"),
        limit: int = Query(50, ge=1, le=200, description="Número máximo de resultados")
):
    """
    Obtener detecciones filtradas por clase de enfermedad.

    Clases disponibles:
    - healthy_cane (HC)
    - leaf_scald (LS)
    - mosaic_virus (MV)
    - painted_fly (PF)
    - red_rot (RR)
    - roya (R)
    - yellow (Y)
    """
    try:
        service = get_history_service()

        if not service.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Servicio de historial no disponible"
            )

        detections = service.get_detections_by_class(class_name=class_name, limit=limit)

        if not detections:
            return []

        # Convertir a modelo de respuesta
        response_data = []
        for detection in detections:
            response_data.append(DetectionHistoryResponse(
                id=detection["_id"],
                timestamp=detection["timestamp"],
                prediction=detection["prediction"],
                processing_info=detection.get("processing_info", {}),
                user_info=detection.get("user_info", {}),
                metadata=detection.get("metadata", {}),
                has_image=detection.get("image", {}).get("has_image", False)
            ))

        return response_data

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al filtrar detecciones por clase: {str(e)}"
        )


@history_router.get("/statistics", response_model=StatisticsResponse)
async def get_detection_statistics():
    """
    Obtener estadísticas del historial de detecciones.

    Incluye:
    - Total de detecciones
    - Distribución por clase de enfermedad
    - Detecciones diarias de los últimos 7 días
    """
    try:
        service = get_history_service()

        if not service.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Servicio de historial no disponible"
            )

        stats = service.get_statistics()

        return StatisticsResponse(
            total_detections=stats.get("total_detections", 0),
            detections_by_class=stats.get("detections_by_class", []),
            daily_detections_last_7_days=stats.get("daily_detections_last_7_days", []),
            last_updated=stats.get("last_updated", datetime.utcnow().isoformat())
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener estadísticas: {str(e)}"
        )


@history_router.get("/{detection_id}")
async def get_detection_by_id(
        detection_id: str = Path(..., description="ID único de la detección")
):
    """
    Obtener una detección específica por su ID.
    """
    try:
        service = get_history_service()

        if not service.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Servicio de historial no disponible"
            )

        detection = service.get_detection_by_id(detection_id)

        if not detection:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró la detección con ID: {detection_id}"
            )

        # Verificar si incluir imagen base64
        if "image" in detection and "base64" in detection["image"]:
            detection["image"]["has_image"] = True
            # Mantener base64 para esta consulta específica

        return detection

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener detección: {str(e)}"
        )


@history_router.get("/{detection_id}/image")
async def get_detection_image(
        detection_id: str = Path(..., description="ID único de la detección")
):
    """
    Obtener la imagen asociada a una detección específica.
    """
    try:
        service = get_history_service()

        if not service.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Servicio de historial no disponible"
            )

        detection = service.get_detection_by_id(detection_id)

        if not detection:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró la detección con ID: {detection_id}"
            )

        # Verificar si tiene imagen
        if "image" not in detection or "base64" not in detection["image"]:
            raise HTTPException(
                status_code=404,
                detail="La detección no tiene imagen asociada"
            )

        # Decodificar imagen base64
        try:
            image_data = base64.b64decode(detection["image"]["base64"])
            image_stream = io.BytesIO(image_data)

            return StreamingResponse(
                io.BytesIO(image_data),
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": f'attachment; filename="detection_{detection_id}.jpg"'
                }
            )

        except Exception as decode_error:
            raise HTTPException(
                status_code=500,
                detail=f"Error al decodificar imagen: {str(decode_error)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener imagen: {str(e)}"
        )


@history_router.delete("/{detection_id}", response_model=DeleteResponse)
async def delete_detection(
        detection_id: str = Path(..., description="ID único de la detección a eliminar")
):
    """
    Eliminar una detección específica del historial.
    """
    try:
        service = get_history_service()

        if not service.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Servicio de historial no disponible"
            )

        success = service.delete_detection(detection_id)

        if success:
            return DeleteResponse(
                success=True,
                message=f"Detección {detection_id} eliminada exitosamente"
            )
        else:
            return DeleteResponse(
                success=False,
                message=f"No se pudo eliminar la detección {detection_id}. Verifique que el ID existe."
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al eliminar detección: {str(e)}"
        )


@history_router.get("/health/check")
async def health_check_history():
    """
    Verificar el estado del servicio de historial.
    """
    try:
        service = get_history_service()

        if service.is_connected():
            return {
                "status": "ok",
                "service": "history",
                "database": "mongodb",
                "connection": "active",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "error",
                "service": "history",
                "database": "mongodb",
                "connection": "inactive",
                "message": "No se puede conectar a MongoDB",
                "timestamp": datetime.utcnow().isoformat()
            }

    except Exception as e:
        return {
            "status": "error",
            "service": "history",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Agregar estos endpoints al final de tu history_endpoints.py

@history_router.get("/user/{user_id}", response_model=List[DetectionHistoryResponse])
async def get_detections_by_user(
        user_id: int = Path(..., description="ID del usuario"),
        limit: int = Query(50, ge=1, le=200, description="Número máximo de resultados"),
        class_filter: Optional[str] = Query(None, description="Filtrar por clase de enfermedad")
):
    """
    Obtener detecciones filtradas por usuario específico.

    Parámetros:
    - user_id: ID del usuario del cual obtener las detecciones
    - limit: Número máximo de resultados a retornar
    - class_filter: Filtrar por clase específica de enfermedad (opcional)
    """
    try:
        service = get_history_service()

        if not service.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Servicio de historial no disponible"
            )

        # Construir filtro de consulta
        query_filter = {"user_info.user.id": user_id}

        if class_filter:
            query_filter["prediction.class_name"] = class_filter

        # Realizar consulta
        cursor = service.collection.find(query_filter).sort("timestamp", -1).limit(limit)

        detections = []
        for doc in cursor:
            doc["timestamp"] = doc["timestamp"].isoformat()
            if "image" in doc and "base64" in doc["image"]:
                doc["image"]["has_image"] = True
                del doc["image"]["base64"]
            detections.append(doc)

        # Convertir a modelo de respuesta
        response_data = []
        for detection in detections:
            response_data.append(DetectionHistoryResponse(
                id=detection["_id"],
                timestamp=detection["timestamp"],
                prediction=detection["prediction"],
                processing_info=detection.get("processing_info", {}),
                user_info=detection.get("user_info", {}),
                metadata=detection.get("metadata", {}),
                has_image=detection.get("image", {}).get("has_image", False)
            ))

        return response_data

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener detecciones del usuario: {str(e)}"
        )


@history_router.get("/user/{user_id}/statistics")
async def get_user_statistics(
        user_id: int = Path(..., description="ID del usuario")
):
    """
    Obtener estadísticas específicas de un usuario.

    Incluye:
    - Total de detecciones realizadas
    - Distribución por clase de enfermedad
    - Actividad reciente (últimos 7 días)
    - Información del usuario
    """
    try:
        service = get_history_service()

        if not service.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Servicio de historial no disponible"
            )

        # Contar total de detecciones del usuario
        total_detections = service.collection.count_documents({"user_info.user.id": user_id})

        if total_detections == 0:
            return {
                "user_id": user_id,
                "total_detections": 0,
                "detections_by_class": [],
                "recent_activity": [],
                "user_info": None,
                "message": "No se encontraron detecciones para este usuario"
            }

        # Estadísticas por clase para el usuario
        pipeline = [
            {"$match": {"user_info.user.id": user_id}},
            {
                "$group": {
                    "_id": "$prediction.class_name",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$prediction.confidence"},
                    "last_detection": {"$max": "$timestamp"},
                    "locations": {"$addToSet": "$user_info.location"}
                }
            },
            {"$sort": {"count": -1}}
        ]

        class_stats = list(service.collection.aggregate(pipeline))

        # Actividad reciente (últimos 7 días)
        from datetime import timedelta
        seven_days_ago = datetime.utcnow() - timedelta(days=7)

        daily_pipeline = [
            {"$match": {
                "user_info.user.id": user_id,
                "timestamp": {"$gte": seven_days_ago}
            }},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$timestamp"
                        }
                    },
                    "count": {"$sum": 1},
                    "classes_detected": {"$addToSet": "$prediction.class_name"}
                }
            },
            {"$sort": {"_id": 1}}
        ]

        daily_stats = list(service.collection.aggregate(daily_pipeline))

        # Obtener información del usuario de la detección más reciente
        recent_detection = service.collection.find_one(
            {"user_info.user.id": user_id},
            sort=[("timestamp", -1)]
        )

        user_info = None
        if recent_detection and "user_info" in recent_detection and "user" in recent_detection["user_info"]:
            user_info = recent_detection["user_info"]["user"]

        # Estadísticas adicionales
        most_common_location = None
        location_pipeline = [
            {"$match": {"user_info.user.id": user_id, "user_info.location": {"$ne": None}}},
            {"$group": {"_id": "$user_info.location", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 1}
        ]

        location_result = list(service.collection.aggregate(location_pipeline))
        if location_result:
            most_common_location = location_result[0]["_id"]

        return {
            "user_id": user_id,
            "user_info": user_info,
            "total_detections": total_detections,
            "detections_by_class": class_stats,
            "recent_activity_last_7_days": daily_stats,
            "most_common_location": most_common_location,
            "summary": {
                "total_classes_detected": len(class_stats),
                "most_detected_class": class_stats[0]["_id"] if class_stats else None,
                "avg_confidence_overall": sum(stat["avg_confidence"] for stat in class_stats) / len(
                    class_stats) if class_stats else 0
            },
            "last_updated": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener estadísticas del usuario: {str(e)}"
        )


@history_router.get("/authenticated", response_model=List[DetectionHistoryResponse])
async def get_authenticated_detections(
        limit: int = Query(50, ge=1, le=200, description="Número máximo de resultados"),
        role_filter: Optional[str] = Query(None, description="Filtrar por rol de usuario")
):
    """
    Obtener solo las detecciones de usuarios autenticados.

    Parámetros:
    - limit: Número máximo de resultados
    - role_filter: Filtrar por rol específico (ADMIN, USER, etc.)
    """
    try:
        service = get_history_service()

        if not service.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Servicio de historial no disponible"
            )

        # Construir filtro
        query_filter = {"user_info.is_authenticated": True}

        if role_filter:
            query_filter["user_info.user.role_names"] = role_filter

        # Filtrar solo detecciones autenticadas
        cursor = service.collection.find(query_filter).sort("timestamp", -1).limit(limit)

        detections = []
        for doc in cursor:
            doc["timestamp"] = doc["timestamp"].isoformat()
            if "image" in doc and "base64" in doc["image"]:
                doc["image"]["has_image"] = True
                del doc["image"]["base64"]
            detections.append(doc)

        # Convertir a modelo de respuesta
        response_data = []
        for detection in detections:
            response_data.append(DetectionHistoryResponse(
                id=detection["_id"],
                timestamp=detection["timestamp"],
                prediction=detection["prediction"],
                processing_info=detection.get("processing_info", {}),
                user_info=detection.get("user_info", {}),
                metadata=detection.get("metadata", {}),
                has_image=detection.get("image", {}).get("has_image", False)
            ))

        return response_data

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener detecciones autenticadas: {str(e)}"
        )


@history_router.get("/users/summary")
async def get_users_summary():
    """
    Obtener un resumen de todos los usuarios que han realizado detecciones.
    """
    try:
        service = get_history_service()

        if not service.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Servicio de historial no disponible"
            )

        # Obtener resumen de usuarios autenticados
        pipeline = [
            {"$match": {"user_info.is_authenticated": True}},
            {
                "$group": {
                    "_id": "$user_info.user.id",
                    "user_email": {"$first": "$user_info.user.email"},
                    "user_name": {"$first": "$user_info.user.fullName"},
                    "total_detections": {"$sum": 1},
                    "last_detection": {"$max": "$timestamp"},
                    "roles": {"$first": "$user_info.user.role_names"},
                    "unique_classes": {"$addToSet": "$prediction.class_name"},
                    "avg_confidence": {"$avg": "$prediction.confidence"}
                }
            },
            {"$sort": {"total_detections": -1}}
        ]

        users_summary = list(service.collection.aggregate(pipeline))

        # Contar detecciones anónimas
        anonymous_count = service.collection.count_documents({"user_info.is_authenticated": False})

        # Estadísticas generales
        total_users = len(users_summary)
        total_authenticated_detections = sum(user["total_detections"] for user in users_summary)

        return {
            "total_authenticated_users": total_users,
            "total_authenticated_detections": total_authenticated_detections,
            "total_anonymous_detections": anonymous_count,
            "users": [
                {
                    "user_id": user["_id"],
                    "email": user["user_email"],
                    "name": user["user_name"],
                    "total_detections": user["total_detections"],
                    "last_detection": user["last_detection"].isoformat() if user["last_detection"] else None,
                    "roles": user["roles"],
                    "unique_classes_detected": len(user["unique_classes"]),
                    "avg_confidence": round(user["avg_confidence"], 3)
                }
                for user in users_summary
            ],
            "last_updated": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener resumen de usuarios: {str(e)}"
        )