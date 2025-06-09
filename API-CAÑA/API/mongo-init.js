// Script de inicialización para MongoDB
// Este archivo se ejecuta automáticamente cuando se crea el contenedor

// Cambiar a la base de datos sugar_cane_db
db = db.getSiblingDB('sugar_cane_db');

// Crear usuario para la aplicación
db.createUser({
  user: 'sugar_cane_user',
  pwd: 'sugar_cane_password',
  roles: [
    {
      role: 'readWrite',
      db: 'sugar_cane_db'
    }
  ]
});

// Crear colección con validación de esquema para detecciones
db.createCollection('detection_history', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['timestamp', 'prediction', 'image', 'metadata'],
      properties: {
        _id: {
          bsonType: 'string',
          description: 'ID único de la detección'
        },
        timestamp: {
          bsonType: 'date',
          description: 'Fecha y hora de la detección'
        },
        prediction: {
          bsonType: 'object',
          required: ['class_id', 'class_name', 'confidence'],
          properties: {
            class_id: {
              bsonType: 'int',
              minimum: 0,
              maximum: 6,
              description: 'ID de la clase detectada (0-6)'
            },
            class_name: {
              bsonType: 'string',
              description: 'Nombre de la enfermedad detectada'
            },
            confidence: {
              bsonType: 'double',
              minimum: 0,
              maximum: 1,
              description: 'Nivel de confianza de la predicción'
            },
            all_predictions: {
              bsonType: 'object',
              description: 'Probabilidades de todas las clases'
            }
          }
        },
        image: {
          bsonType: 'object',
          required: ['base64', 'original_size'],
          properties: {
            base64: {
              bsonType: 'string',
              description: 'Imagen codificada en base64'
            },
            original_size: {
              bsonType: 'array',
              description: 'Tamaño original de la imagen [width, height]'
            },
            format: {
              bsonType: 'string',
              description: 'Formato de la imagen'
            }
          }
        },
        processing_info: {
          bsonType: 'object',
          description: 'Información del procesamiento aplicado'
        },
        user_info: {
          bsonType: 'object',
          description: 'Información del usuario y contexto'
        },
        metadata: {
          bsonType: 'object',
          required: ['api_version', 'model_type'],
          properties: {
            api_version: {
              bsonType: 'string',
              description: 'Versión de la API'
            },
            model_type: {
              bsonType: 'string',
              description: 'Tipo de modelo usado'
            },
            created_at: {
              bsonType: 'string',
              description: 'Timestamp de creación en formato ISO'
            }
          }
        }
      }
    }
  }
});

// Crear índices para optimizar consultas
db.detection_history.createIndex({ "timestamp": -1 }, { name: "timestamp_desc" });
db.detection_history.createIndex({ "prediction.class_name": 1 }, { name: "class_name_asc" });
db.detection_history.createIndex({ "prediction.confidence": -1 }, { name: "confidence_desc" });
db.detection_history.createIndex({ 
  "timestamp": -1, 
  "prediction.class_name": 1 
}, { name: "timestamp_class_compound" });

// Crear índice para información de usuario (opcional)
db.detection_history.createIndex({ "user_info.user_id": 1 }, { name: "user_id_asc" });

print('✅ Base de datos sugar_cane_db inicializada correctamente');
print('✅ Usuario sugar_cane_user creado');
print('✅ Colección detection_history creada con validación');
print('✅ Índices creados para optimizar consultas');
