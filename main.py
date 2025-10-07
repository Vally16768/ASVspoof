# main.py

import os
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from build_CRNN_model import build_model
from constants import epochs, batch_size
import datetime
from utilities import save_best_combination
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import Counter
from sklearn.utils import class_weight
import h5py
# def evaluate_best_model(X_test, y_test, label_encoder):
#     # Load the best saved model
#     best_model = load_model('best_model.h5')
    
#     # Evaluate on the test data
#     test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
#     print(f"Best model test loss: {test_loss:.4f}")
#     print(f"Best model test accuracy: {test_accuracy:.4f}")

#     # Generate predictions
#     y_pred = best_model.predict(X_test)
#     y_pred_classes = np.argmax(y_pred, axis=1)

#     # Create confusion matrix
#     cm = confusion_matrix(y_test, y_pred_classes)
    
#     # Plot confusion matrix
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix for Best Model')
#     plt.savefig('temp_data_folder/best_model_confusion_matrix.jpg')
#     plt.show()
    
#     # Classification report
#     report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
#     print("Classification Report:\n", report)
#     # Save classification report
#     with open('temp_data_folder/best_model_classification_report.txt', 'w') as f:
#         f.write(report)
#     print("Classification report saved to temp_data_folder/best_model_classification_report.txt")

def normalize_data(X, scaler=None):
    n_samples, n_time_steps, n_features = X.shape
    X_flat = X.reshape(n_samples * n_time_steps, n_features)
    if scaler is None:
        scaler = StandardScaler()
        X_flat = scaler.fit_transform(X_flat)
    else:
        X_flat = scaler.transform(X_flat)
    X_normalized = X_flat.reshape(n_samples, n_time_steps, n_features)
    return X_normalized, scaler

def plot_history(history, save_dir='temp_data_folder'):
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    plt.figure(figsize=(12, 4))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label='Train Loss')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_history.jpg')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to {plot_path}")

def main():
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Load datasets
    data_dir = 'dataset/extracted_features'
    if (os.path.exists(os.path.join(data_dir, 'X_train.npy')) and
        os.path.exists(os.path.join(data_dir, 'X_val.npy')) and
        os.path.exists(os.path.join(data_dir, 'X_test.npy')) and
        os.path.exists(os.path.join(data_dir, 'y_train.npy')) and
        os.path.exists(os.path.join(data_dir, 'y_val.npy')) and
        os.path.exists(os.path.join(data_dir, 'y_test.npy')) and
        os.path.exists(os.path.join(data_dir, 'label_encoder.joblib'))):
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        le = joblib.load(os.path.join(data_dir, 'label_encoder.joblib'))
        print("Datasets loaded successfully.")
    else:
        print("Datasets not found. Please run 'create_dataset.py' first to generate the datasets.")
        return
    
    # Verify data shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Normalize datasets using a single scaler
    X_train, scaler = normalize_data(X_train)
    X_val, _ = normalize_data(X_val, scaler)
    X_test, _ = normalize_data(X_test, scaler)
    print(f"NaNs in X_train after normalization: {np.isnan(X_train).sum()}")
    print(f"Infs in X_train after normalization: {np.isinf(X_train).sum()}")
    
    # Reshape data for CNN input
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    n_time_steps, n_features = X_train.shape[1], X_train.shape[2]
    n_classes = len(le.classes_)
    print("Encoded classes:", le.classes_)
    print("Sample features shape:", X_train[0].shape)
    print("Sample label:", y_train[0])
    print("n_time_steps:", n_time_steps)
    print("n_features:", n_features)
    print("n_classes:", n_classes)

    # Build model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model = build_model(n_time_steps, n_features, n_classes)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True
    )

    model_path = 'best_model.h5'

    # If the file exists, delete it to avoid conflicts
    if os.path.exists(model_path):
        # Close any open handles on the file
        with h5py.File(model_path, 'r+') as f:
            f.close()
        
        # Delete the file to ensure we can create a new one
        os.remove(model_path)
        print(f"Deleted existing model file: {model_path}")

    callbacks = [early_stopping, reduce_lr, tensorboard_callback]

    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    # Visualize sample data
    if not os.path.exists('temp_data_folder'):
        os.makedirs('temp_data_folder')

    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Plot training history
    plot_history(history, save_dir='temp_data_folder')

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')
    # evaluate_best_model(X_test, y_test,le)
    # Save best combination
    save_best_combination(test_accuracy, 'improved_model')

    # Generate confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('temp_data_folder/confusion_matrix.jpg')
    plt.close()
    print("Confusion matrix saved to temp_data_folder/confusion_matrix.jpg")

    # Classification report
    report = classification_report(y_test, y_pred_classes, target_names=le.classes_)
    print("Classification Report:\n", report)
    # Save classification report
    with open('temp_data_folder/classification_report.txt', 'w') as f:
        f.write(report)
    print("Classification report saved to temp_data_folder/classification_report.txt")
    print("Training set class distribution:", Counter(y_train))
    
    plt.bar(range(n_classes), [Counter(y_train)[i] for i in range(n_classes)])
    plt.xticks(range(n_classes), le.classes_)
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.savefig('temp_data_folder/class_distribution.jpg')
    plt.close()

if __name__ == '__main__':
    main()
