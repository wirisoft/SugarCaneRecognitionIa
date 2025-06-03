import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import densenet161, DenseNet161_Weights
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import time
import random
import joblib
import gc
import warnings
import pandas as pd
from tqdm import tqdm
import multiprocessing

# Configuración de CPU
NUM_CPUS = multiprocessing.cpu_count()
NUM_WORKERS = min(32, NUM_CPUS)  # Usar hasta 32 workers
torch.set_num_threads(NUM_CPUS)  # Configurar número de hilos para operaciones CPU

# Filtrar advertencias
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".GradScaler.")
warnings.filterwarnings("ignore", message=".autocast.")
warnings.filterwarnings("ignore", message="torch.load is now *")
warnings.filterwarnings("ignore", message=".The default behavior for.")


# Configuración de semillas para reproducibilidad
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

# Optimizar el uso de memoria CUDA
# Establecer la fracción máxima de memoria GPU que PyTorch puede reservar (1.0 = 100%)
torch.cuda.set_per_process_memory_fraction(1.0)
# Configurar el manejo de memoria fragmentada
torch.cuda.empty_cache()
if hasattr(torch.cuda, 'memory_stats'):
    torch.cuda.memory_stats()  # Recolectar estadísticas de uso
# Configurar comportamiento de memoria compartida
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
# Configurar número de workers para CPU
os.environ['OMP_NUM_THREADS'] = '24'  # Número de núcleos físicos
os.environ['MKL_NUM_THREADS'] = '24'
os.environ['OPENBLAS_NUM_THREADS'] = '24'
os.environ['VECLIB_MAXIMUM_THREADS'] = '24'
os.environ['NUMEXPR_NUM_THREADS'] = '24'

# Verificar si CUDA está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Para diagnóstico de CUDA
if torch.cuda.is_available():
    # Mostrar información de memoria a detalle
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Memoria GPU asignada: {allocated:.2f} GB")
    print(f"Memoria GPU reservada: {reserved:.2f} GB")

# Configuración para precisión mixta (FP16)
USE_MIXED_PRECISION = True
scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)

# Definición de rutas
# Carpeta para guardar todos los modelos y resultados
OUTPUT_DIR = r"C:\Users\misju\SugarCaneRecognitionIa\rpeca\MODELS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_DIR = r"C:\Users\misju\SugarCaneRecognitionIa\rpeca\hojas_cania_dataset"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "MODELS")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "RESULTS")

# Crear directorios si no existen
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "MODELS")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "RESULTS")

# Crear directorios si no existen
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)



# Clase para el dataset de imágenes
class SugarCaneDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True, train_ratio=0.8):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.is_train = is_train


        # Mapeo de enfermedades a etiquetas
        self.disease_mapping = {
            'healthy_cane (HC)' : 0,
            'leaf_scald (LS)': 1,
            'mosaic_virus (MV)': 2,
            'painted_fly (PF)': 3,
            'red_rot (RR)': 4,
            'roya (R)': 5,
            'yellow (Y)': 6,
        }

        # Cargar imágenes de cada enfermedad
        for disease_folder, label in self.disease_mapping.items():
            disease_path = os.path.join(root_dir, disease_folder)
            if os.path.exists(disease_path):
                # Obtener lista de todas las imágenes
                all_images = [img for img in os.listdir(disease_path)
                              if img.endswith(('.jpg', '.jpeg', '.png'))]

                # Calcular índices de división
                n_images = len(all_images)
                n_train = int(n_images * train_ratio)

                # Dividir en train y test
                if is_train:
                    selected_images = all_images[:n_train]
                else:
                    selected_images = all_images[n_train:]

                # Agregar las imágenes seleccionadas al dataset
                for img_name in selected_images:
                    self.samples.append((os.path.join(disease_path, img_name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# Transformaciones para preprocesamiento de imágenes
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Tamaño requerido para DenseNet-161
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Tamaño requerido para DenseNet-161
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inicialización de datasets y dataloaders
train_dataset = SugarCaneDataset(BASE_DIR, transform=train_transform, is_train=True)
test_dataset = SugarCaneDataset(BASE_DIR, transform=test_transform, is_train=False)

# Contar imágenes por clase en train y test
train_class_counts = {}
test_class_counts = {}

for _, label in train_dataset.samples:
    train_class_counts[label] = train_class_counts.get(label, 0) + 1

for _, label in test_dataset.samples:
    test_class_counts[label] = test_class_counts.get(label, 0) + 1

print("\nConteo de imágenes por clase en entrenamiento:")
for label, count in sorted(train_class_counts.items()):
    disease_name = [k for k, v in train_dataset.disease_mapping.items() if v == label][0]
    print(f"{disease_name}: {count}")

print("\nConteo de imágenes por clase en prueba:")
for label, count in sorted(test_class_counts.items()):
    disease_name = [k for k, v in test_dataset.disease_mapping.items() if v == label][0]
    print(f"{disease_name}: {count}")

# Batch size ajustado para DenseNet-161
BATCH_SIZE = 8  # Reducido para manejar mejor la memoria
NUM_WORKERS = 8  # Reducido para evitar problemas de memoria
ACCUMULATION_STEPS = 32  # Aumentado para compensar el batch size más pequeño
NUM_EPOCHS = 50

# Usar el dataset sin aumento para el entrenamiento y test
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)


# Clase para el modelo DenseNet-161
class ODIRDenseNetModel(nn.Module):
    def __init__(self, freeze_backbone=True, initial_unfreeze_level=0):
        super(ODIRDenseNetModel, self).__init__()

        # Cargar modelo DenseNet-161 pre-entrenado
        self.densenet = densenet161(weights=DenseNet161_Weights.DEFAULT)

        # Congelar todo inicialmente si se especifica
        if freeze_backbone:
            for param in self.densenet.parameters():
                param.requires_grad = False

        # Modificar la cabeza del DenseNet para clasificación multiclase
        densenet_output_dim = self.densenet.classifier.in_features

        # Reemplazar la cabeza original con un nuevo clasificador
        self.densenet.classifier = nn.Sequential(
            nn.Linear(densenet_output_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 7)  # 7 clases para las diferentes enfermedades
        )

        # Aplicar nivel de descongelamiento inicial si se especifica
        if freeze_backbone and initial_unfreeze_level > 0:
            self.apply_initial_unfreezing(initial_unfreeze_level)

        # Asegurar que el nivel de descongelamiento se mantenga constante
        self._freeze_state = initial_unfreeze_level
        self._frozen_params = set()
        self._update_frozen_params()

        # Registrar el estado inicial de congelamiento
        self._initial_freeze_state = self._get_freeze_state()

    def _get_freeze_state(self):
        """Obtiene el estado actual de congelamiento de todos los parámetros"""
        return {name: param.requires_grad for name, param in self.named_parameters()}

    def _update_frozen_params(self):
        """Actualiza el conjunto de parámetros congelados"""
        self._frozen_params.clear()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                self._frozen_params.add(name)

    def _ensure_freeze_state(self):
        """Asegura que el estado de congelamiento se mantenga constante"""
        for name, param in self.named_parameters():
            if name in self._frozen_params:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Verificar que el estado de congelamiento no haya cambiado
        current_state = self._get_freeze_state()
        if current_state != self._initial_freeze_state:
            raise RuntimeError("El estado de congelamiento ha cambiado durante el entrenamiento")

    def apply_initial_unfreezing(self, level):
        """
        Aplica un nivel inicial de descongelamiento al modelo DenseNet-161.

        Args:
            level: Nivel de descongelamiento
                1=leve (~5%)
                2=medio (~10%)
                2.5=intermedio (~50%)
                3=alto (~75%)
                4=completo (100%)
        """
        if level >= 1:  # Nivel leve (últimas capas) ~5%
            print("Aplicando descongelamiento inicial: nivel leve (últimas capas ~5%)")
            for name, param in self.densenet.named_parameters():
                if 'denseblock4' in name or 'classifier' in name:
                    param.requires_grad = True

        if level >= 2:  # Nivel medio (capas intermedias) ~10%
            print("Aplicando descongelamiento inicial: nivel medio (capas intermedias ~10%)")
            for name, param in self.densenet.named_parameters():
                if 'denseblock3' in name or 'denseblock4' in name or 'classifier' in name:
                    param.requires_grad = True

        if level >= 2.5:  # Nivel intermedio (~50% de los parámetros)
            print("Aplicando descongelamiento inicial: nivel intermedio (aproximadamente 50% del modelo)")
            for name, param in self.densenet.named_parameters():
                if 'denseblock2' in name or 'denseblock3' in name or 'denseblock4' in name or 'classifier' in name:
                    param.requires_grad = True

        if level >= 3:  # Nivel alto (la mayoría de las capas) ~75%
            print("Aplicando descongelamiento inicial: nivel alto (la mayoría de capas ~75%)")
            for name, param in self.densenet.named_parameters():
                if 'denseblock1' in name or 'denseblock2' in name or 'denseblock3' in name or 'denseblock4' in name or 'classifier' in name:
                    param.requires_grad = True

        if level >= 4:  # Descongelamiento completo (100%)
            print("Aplicando descongelamiento completo (100% de los parámetros)")
            for param in self.parameters():
                param.requires_grad = True

        # Mostrar estadísticas del descongelamiento inicial
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"Estado inicial - Parámetros entrenables: {trainable_params:,} de {total_params:,} ({trainable_params / total_params * 100:.2f}%)")

    def forward(self, x):
        return self.densenet(x)


# Modificar la función de pérdida para clasificación multiclase
def weighted_cross_entropy(outputs, targets):
    # Asegurar que los targets sean 1D
    targets = targets.view(-1)

    # Calcular pesos para cada clase basado en la frecuencia
    class_counts = torch.bincount(targets.long(), minlength=7)
    total_samples = len(targets)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)

    loss = nn.CrossEntropyLoss(weight=class_weights)
    return loss(outputs, targets.long())


# Modificar la función de entrenamiento para clasificación multiclase
def train_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=1):
    model.train()
    model._ensure_freeze_state()

    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Entrenando", unit="batch")

    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Usar autocast para precisión mixta
        with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss = loss / accumulation_steps

        # Escalar y propagar el gradiente
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Actualizar parámetros
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            model._ensure_freeze_state()

        running_loss += loss.item() * inputs.size(0) * accumulation_steps
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss / total:.4f}',
            'acc': f'{correct / total:.4f}'
        })

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, find_threshold=False, is_train=False):
    if is_train:
        mode = model.training
        model.eval()
    else:
        model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    # Obtener el disease_mapping del dataset
    disease_mapping = dataloader.dataset.disease_mapping

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluando" if not is_train else "Evaluando Train"):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast('cuda', enabled=USE_MIXED_PRECISION):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            torch.cuda.empty_cache()

    if is_train and mode:
        model.train()

    val_loss = running_loss / len(dataloader.dataset)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    val_acc = np.mean(all_preds == all_labels)

    # Convertir etiquetas a formato de clase única
    all_labels = all_labels.astype(int)
    all_preds = all_preds.astype(int)

    report = classification_report(all_labels, all_preds, output_dict=True)

    # Calcular métricas por clase
    metrics = {}
    for i in range(7):  # 7 clases
        class_name = [k for k, v in disease_mapping.items() if v == i][0]
        metrics[f'precision_{class_name}'] = report[str(i)]['precision']
        metrics[f'recall_{class_name}'] = report[str(i)]['recall']
        metrics[f'f1_{class_name}'] = report[str(i)]['f1-score']

    metrics['accuracy'] = report['accuracy']
    metrics['macro_avg_precision'] = report['macro avg']['precision']
    metrics['macro_avg_recall'] = report['macro avg']['recall']
    metrics['macro_avg_f1'] = report['macro avg']['f1-score']

    try:
        # Calcular AUC para cada clase
        auc_scores = []
        for i in range(7):
            auc = roc_auc_score((all_labels == i).astype(int), all_probs[:, i])
            auc_scores.append(auc)
        metrics['auc'] = np.mean(auc_scores)
    except:
        metrics['auc'] = 0.0

    return val_loss, val_acc, report, metrics['auc'], all_labels, all_preds, all_probs, None, metrics


# Función para entrenar el modelo con métricas adicionales
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=20, accumulation_steps=1):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': [],
        'train_auc': []
    }

    best_val_auc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps)

        torch.cuda.empty_cache()
        gc.collect()

        # Evaluar en conjunto de entrenamiento
        _, _, _, train_auc, _, _, _, _, train_metrics = validate(model, train_loader, criterion, device, is_train=True)

        # Evaluar en conjunto de validación
        val_loss, val_acc, report, val_auc, _, _, _, _, metrics = validate(model, test_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "best_densenet161_model.pth"))
            print(f"Mejor modelo guardado con AUC: {val_auc:.4f}")

        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch + 1}.pth"))
        print(f"Modelo de época {epoch + 1} guardado")

        # Guardar todas las métricas
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_precision'].append(train_metrics['macro_avg_precision'])
        history['train_recall'].append(train_metrics['macro_avg_recall'])
        history['train_f1'].append(train_metrics['macro_avg_f1'])
        history['train_auc'].append(train_metrics['auc'])

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_precision'].append(metrics['macro_avg_precision'])
        history['val_recall'].append(metrics['macro_avg_recall'])
        history['val_f1'].append(metrics['macro_avg_f1'])

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        print(
            f"Precision - Train: {train_metrics['macro_avg_precision']:.4f}, Val: {metrics['macro_avg_precision']:.4f}")
        print(f"Recall - Train: {train_metrics['macro_avg_recall']:.4f}, Val: {metrics['macro_avg_recall']:.4f}")
        print(f"F1-Score - Train: {train_metrics['macro_avg_f1']:.4f}, Val: {metrics['macro_avg_f1']:.4f}")
        print(f"AUC - Train: {train_metrics['auc']:.4f}, Val: {val_auc:.4f}")

        # Guardar checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_val_auc': best_val_auc,
            'history': history
        }
        torch.save(checkpoint, os.path.join(MODEL_SAVE_PATH, "checkpoint.pth"))

        # Guardar historial en CSV
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(RESULTS_PATH, "training_history.csv"), index=False)

        torch.cuda.empty_cache()
        gc.collect()

    time_elapsed = time.time() - start_time
    print(f'\nEntrenamiento completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Mejor AUC de validación: {best_val_auc:.4f}')

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(RESULTS_PATH, "training_history_final.csv"), index=False)

    return history, None


# Función para evaluar el modelo con las nuevas métricas
def evaluate_model(model_path, test_loader, device, threshold=0.5):
    model = ODIRDenseNetModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)

    criterion = weighted_cross_entropy
    _, _, report, auc, all_labels, all_preds, all_probs, _, metrics = validate(model, test_loader, criterion, device)

    print(f"\nUmbral fijo utilizado: {threshold:.4f}")
    print("\nInforme de Clasificación:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Sensibilidad (Recall Clase 1): {report['1.0']['recall']:.4f}")
    print(f"Especificidad (Recall Clase 0): {report['0.0']['recall']:.4f}")
    print(f"Precisión (Clase 1): {report['1.0']['precision']:.4f}")
    print(f"F1-Score (Clase 1): {report['1.0']['f1-score']:.4f}")

    return model, report, auc, all_labels, all_preds, all_probs, threshold, metrics


# Función principal para ejecutar el entrenamiento y evaluación
def main():
    print("\nConfigurando optimizaciones de memoria y CPU...")
    if USE_MIXED_PRECISION:
        print("- Precisión mixta (FP16) activada")
    else:
        print("- Usando precisión completa (FP32)")

    print(f"- Fracción máxima de memoria GPU: 100%")
    print(f"- Tamaño máximo de split de memoria: 128 MB")
    print(f"- Número de workers para DataLoader: {NUM_WORKERS}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Pasos de acumulación: {ACCUMULATION_STEPS}")

    gc.collect()
    torch.cuda.empty_cache()

    # Inicializar modelo con nivel de descongelamiento completo (100% de los parámetros)
    model = ODIRDenseNetModel(freeze_backbone=True, initial_unfreeze_level=4)
    model = model.to(device)

    criterion = weighted_cross_entropy
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Inicializar scheduler ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Queremos minimizar la val_loss
        factor=0.5,  # Reduce el LR a la mitad
        patience=3,  # Espera 3 épocas sin mejora
        verbose=True
    )

    config = {
        'batch_size': BATCH_SIZE,
        'accumulation_steps': ACCUMULATION_STEPS,
        'num_workers': NUM_WORKERS,
        'optimizer': 'AdamW',
        'learning_rate': 0.0001,
        'weight_decay': 1e-4,
        'scheduler': 'ReduceLROnPlateau',
        'num_epochs': NUM_EPOCHS,
        'freeze_backbone': True,
        'adaptive_unfreezing': False,
        'initial_unfreeze_level': 4,
        'use_mixed_precision': USE_MIXED_PRECISION,
        'cuda_memory_fraction': 1.0,
        'cuda_max_split_size_mb': 128,
        'model': 'DenseNet-161',
        'num_classes': 7,
        'image_size': 224
    }
    pd.DataFrame([config]).to_csv(os.path.join(RESULTS_PATH, "config.csv"), index=False)

    print("Iniciando entrenamiento del modelo DenseNet-161...")
    history, _ = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                             num_epochs=NUM_EPOCHS, accumulation_steps=ACCUMULATION_STEPS)

    print("\nEvaluando el mejor modelo en el conjunto de prueba...")
    best_model_path = os.path.join(MODEL_SAVE_PATH, "best_densenet161_model.pth")
    model, report, auc, labels, preds, probs, final_threshold, metrics = evaluate_model(best_model_path, test_loader,
                                                                                        device, 0.5)

    print(f"\nTodos los resultados guardados en: {OUTPUT_DIR}")

    return model


if __name__ == "__main__":
    main()