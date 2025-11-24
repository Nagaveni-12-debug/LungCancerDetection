import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class CancerModelTrainer:
    def __init__(self, data_dir='data/', model_save_dir='models/best_modelsIS/'):
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.model = None
        self.history = None
        
        # Create directories if they don't exist
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs('models/img_cancer_modelsIS/', exist_ok=True)
    
    def load_and_preprocess_data(self):
        """Load and preprocess the cancer dataset"""
        print("Loading and preprocessing data...")
        
        # Expected directory structure:
        # data/
        #   cancer/
        #     image1.jpg, image2.png, ...
        #   healthy/
        #     image1.jpg, image2.png, ...
        
        images = []
        labels = []
        
        # Define classes
        classes = ['healthy', 'cancer']
        
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist. Skipping...")
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, filename)
                    
                    try:
                        # Load and preprocess image
                        img = Image.open(img_path)
                        img = img.convert('RGB')
                        img = img.resize((224, 224))
                        img_array = np.array(img) / 255.0
                        
                        images.append(img_array)
                        labels.append(class_idx)
                        
                    except Exception as e:
                        print(f"Error loading {img_path}: {str(e)}")
                        continue
        
        if len(images) == 0:
            raise Exception("No images found in the data directory!")
        
        print(f"Loaded {len(images)} images")
        return np.array(images), np.array(labels)
    
    def create_model(self, input_shape=(224, 224, 3), num_classes=2):
        """Create a CNN model for cancer detection"""
        print("Creating model architecture...")
        
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def train_model(self, epochs=50, batch_size=32, validation_split=0.2):
        """Train the cancer detection model"""
        print("Starting model training...")
        
        # Load data
        X, y = self.load_and_preprocess_data()
        
        # Convert labels to categorical
        y_categorical = keras.utils.to_categorical(y, num_classes=2)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        
        # Create model
        self.model = self.create_model()
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_save_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
        return self.history
    
    def evaluate_model(self, X_test=None, y_test=None):
        """Evaluate the trained model"""
        if self.model is None:
            raise Exception("Model not trained yet!")
        
        if X_test is None or y_test is None:
            # Load data for evaluation
            X, y = self.load_and_preprocess_data()
            y_categorical = keras.utils.to_categorical(y, num_classes=2)
            _, X_test, _, y_test = train_test_split(
                X, y_categorical, test_size=0.2, random_state=42, stratify=y
            )
        
        # Evaluate model
        evaluation = self.model.evaluate(X_test, y_test, verbose=0)
        
        print("\nModel Evaluation:")
        print(f"Loss: {evaluation[0]:.4f}")
        print(f"Accuracy: {evaluation[1]:.4f}")
        print(f"Precision: {evaluation[2]:.4f}")
        print(f"Recall: {evaluation[3]:.4f}")
        
        return evaluation
    
    def save_model(self, save_path=None):
        """Save the trained model and architecture"""
        if self.model is None:
            raise Exception("No model to save!")
        
        if save_path is None:
            save_path = os.path.join(self.model_save_dir, 'cancer_model.h5')
        
        # Save model weights
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
        
        # Save model architecture
        model_architecture = self.model.to_json()
        architecture_path = 'models/model_architecture.json'
        with open(architecture_path, 'w') as f:
            json.dump(model_architecture, f)
        print(f"Model architecture saved to {architecture_path}")
        
        # Save training history
        history_path = os.path.join(self.model_save_dir, 'training_history.npy')
        np.save(history_path, self.history.history)
        print(f"Training history saved to {history_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            raise Exception("No training history available!")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_sample_data():
    """Create sample data structure for testing"""
    sample_dir = 'data/'
    os.makedirs(os.path.join(sample_dir, 'healthy'), exist_ok=True)
    os.makedirs(os.path.join(sample_dir, 'cancer'), exist_ok=True)
    print("Sample directory structure created at 'data/'")
    print("Please add your medical images in:")
    print("  - data/healthy/  (for healthy tissue images)")
    print("  - data/cancer/   (for cancerous tissue images)")

# Main execution
if __name__ == "__main__":
    # Create sample directory structure
    create_sample_data()
    
    # Initialize trainer
    trainer = CancerModelTrainer()
    
    try:
        # Train model
        history = trainer.train_model(epochs=50, batch_size=32)
        
        # Evaluate model
        trainer.evaluate_model()
        
        # Save model
        trainer.save_model()
        
        # Plot training history
        trainer.plot_training_history()
        
        print("\n🎉 Model training completed successfully!")
        print("The trained model is ready for use in prediction.")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("\n📁 Please make sure you have placed your medical images in:")
        print("   - data/healthy/  (for healthy tissue images)")
        print("   - data/cancer/   (for cancerous tissue images)")