import os
import numpy as np
import pandas as pd
import nibabel as nib
import mne
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import ParameterGrid
import logging
from datetime import datetime
import mlflow
import mlflow.keras
import cv2
from tensorflow.keras import backend as K

# Configure logging and MLflow
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'neurovr_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
mlflow.set_experiment("NeuroVR_Pipeline")

# Data augmentation for MRI
def get_mri_datagen():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

# Validate and preprocess MRI data
def load_mri_images(mri_folder):
    try:
        mri_images, labels = [], []
        for file in os.listdir(mri_folder):
            if file.endswith((".nii", ".nii.gz")):
                img_path = os.path.join(mri_folder, file)
                img = nib.load(img_path).get_fdata()
                if not validate_mri_data(img):
                    logger.warning(f"Invalid MRI data in {file}")
                    continue
                img = preprocess_mri(img)
                mri_images.append(img)
                labels.append(1 if "hiv" in file.lower() else 0)
        if not mri_images:
            raise ValueError("No valid MRI images found")
        return np.array(mri_images), np.array(labels)
    except Exception as e:
        logger.error(f"Error loading MRI data: {str(e)}")
        raise

def validate_mri_data(img):
    return (img.ndim == 3 and
            all(dim >= 64 for dim in img.shape) and
            not np.any(np.isnan(img)) and
            img.size > 0)

def preprocess_mri(img):
    img = img[:, :, :64]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = np.expand_dims(img, axis=-1)
    return img.astype(np.float32)

# Validate and preprocess EEG data
def load_eeg_edf(eeg_folder):
    try:
        eeg_data, labels = [], []
        for file in os.listdir(eeg_folder):
            if file.endswith(".edf"):
                raw = mne.io.read_raw_edf(os.path.join(eeg_folder, file), preload=True)
                if not validate_eeg_data(raw):
                    logger.warning(f"Invalid EEG data in {file}")
                    continue
                data = preprocess_eeg(raw)
                eeg_data.append(data)
                labels.append(1 if "seizure" in file.lower() else 0)
        if not eeg_data:
            raise ValueError("No valid EEG data found")
        return np.array(eeg_data), np.array(labels)
    except Exception as e:
        logger.error(f"Error loading EEG data: {str(e)}")
        raise

def validate_eeg_data(raw):
    data, _ = raw[:]
    return (data.shape[1] >= 256 and
            not np.any(np.isnan(data)) and
            data.size > 0)

def preprocess_eeg(raw):
    raw.resample(128)
    data, _ = raw[:]
    data = data[:, :256]
    data = (data - data.mean()) / (data.std() + 1e-8)
    return data.T.astype(np.float32)

# Grad-CAM implementation
def get_gradcam_heatmap(model, img_array, layer_name):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    
    return heatmap.numpy()

# Enhanced MRI model
def build_mri_model(input_shape, dropout_rate=0.4):
    inputs = Input(shape=input_shape)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='last_conv')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

# Enhanced EEG model
def build_eeg_model(input_shape, dropout_rate=0.4):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

# Multimodal fusion model
def build_fusion_model(mri_model, eeg_model, fusion_units=128):
    mri_features = mri_model.layers[-2].output
    eeg_features = eeg_model.layers[-2].output
    
    combined = Concatenate()([mri_features, eeg_features])
    x = Dense(fusion_units, activation='relu')(combined)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[mri_model.input, eeg_model.input], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

# Plot training history and confusion matrix
def plot_metrics(history, y_true, y_pred, model_name):
    fig, ((ax1, ax2), (ax3, _)) = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title(f'{model_name} Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title(f'{model_name} Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title(f'{model_name} Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_metrics.png')
    plt.close()

# Hyperparameter tuning
def tune_model(model_builder, X, y, param_grid, model_name):
    best_score, best_params = -float('inf'), None
    best_model = None
    
    for params in ParameterGrid(param_grid):
        logger.info(f"Testing parameters: {params}")
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            model = model_builder(X[0].shape, **params)
            scores = []
            
            kfold = KFold(n_splits=3, shuffle=True, random_state=42)
            for train_idx, val_idx in kfold.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10,
                    batch_size=8,
                    callbacks=[EarlyStopping(patience=3)],
                    verbose=0
                )
                score = max(history.history['val_accuracy'])
                scores.append(score)
            
            avg_score = np.mean(scores)
            mlflow.log_metric("avg_val_accuracy", avg_score)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                best_model = model
    
    logger.info(f"Best parameters for {model_name}: {best_params}")
    return best_model, best_params

# Train and evaluate model
def train_and_evaluate(model, X, y, model_name, input_type='single', is_fusion=False):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ModelCheckpoint(f"{model_name}_best.h5", save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]
    
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X if not is_fusion else X[0], y)):
            logger.info(f"Training fold {fold + 1}/5 for {model_name}")
            
            if is_fusion:
                X_train = [X[0][train_idx], X[1][train_idx]]
                X_val = [X[0][val_idx], X[1][val_idx]]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if input_type == 'mri':
                datagen = get_mri_datagen()
                datagen.fit(X_train)
                flow = datagen.flow(X_train, y_train, batch_size=8)
                history = model.fit(
                    flow,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    steps_per_epoch=len(X_train) // 8,
                    callbacks=callbacks
                )
            else:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=8,
                    callbacks=callbacks
                )
            
            y_pred = (model.predict(X_val) > 0.5).astype("int32")
            plot_metrics(history, y_val, y_pred, f"{model_name}_fold_{fold + 1}")
            
            report = classification_report(y_val, y_pred, output_dict=True)
            fold_scores.append(report)
            
            mlflow.log_metrics({
                f"fold_{fold + 1}_precision": report['1']['precision'],
                f"fold_{fold + 1}_recall": report['1']['recall']
            })
            
            # Generate Grad-CAM for MRI
            if input_type == 'mri':
                sample_img = X_val[0:1]
                heatmap = get_gradcam_heatmap(model, sample_img, 'last_conv')
                np.save(f"{model_name}_gradcam_fold_{fold + 1}.npy", heatmap)
        
        avg_precision = np.mean([score['1']['precision'] for score in fold_scores])
        avg_recall = np.mean([score['1']['recall'] for score in fold_scores])
        mlflow.log_metrics({"avg_precision": avg_precision, "avg_recall": avg_recall})
        
        logger.info(f"{model_name} Average Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}")
        model.save(f"{model_name}_final.h5")
        mlflow.keras.log_model(model, f"{model_name}_model")
    
    return model

# Main pipeline
if __name__ == "__main__":
    MRI_PATH = "/path/to/MRI_dataset"
    EEG LAND = "/path/to/EEG_dataset"
    
    try:
        # Hyperparameter grids
        mri_param_grid = {
            'dropout_rate': [0.3, 0.4, 0.5],
            'learning_rate': [0.001, 0.0005]
        }
        eeg_param_grid = {
            'dropout_rate': [0.3, 0.4, 0.5],
            'learning_rate': [0.001, 0.0005]
        }
        
        # MRI Pipeline
        logger.info("Starting MRI pipeline")
        X_mri, y_mri = load_mri_images(MRI_PATH)
        mri_model, mri_best_params = tune_model(build_mri_model, X_mri, y_mri, mri_param_grid, "NeuroVR_MRI_Model")
        mri_model = train_and_evaluate(mri_model, X_mri, y_mri, "NeuroVR_MRI_Model", input_type='mri')
        
        # EEG Pipeline
        logger.info("Starting EEG pipeline")
        X_eeg, y_eeg = load_eeg_edf(EEG_PATH)
        eeg_model, eeg_best_params = tune_model(build_eeg_model, X_eeg, y_eeg, eeg_param_grid, "NeuroVR_EEG_Model")
        eeg_model = train_and_evaluate(eeg_model, X_eeg, y_eeg, "NeuroVR_EEG_Model", input_type='eeg')
        
        # Fusion Model
        if len(X_mri) == len(X_eeg) and len(y_mri) == len(y_eeg):
            logger.info("Starting fusion model pipeline")
            fusion_model = build_fusion_model(mri_model, eeg_model)
            fusion_model = train_and_evaluate(
                fusion_model,
                [X_mri, X_eeg],
                y_mri,
                "NeuroVR_Fusion_Model",
                input_type='fusion',
                is_fusion=True
            )
        
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
