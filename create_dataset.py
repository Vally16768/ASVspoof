import os
import numpy as np
from prepare_dataset import prepare_dataset
import joblib  # For saving the LabelEncoder

def main():
    save_dir = 'dataset/extracted_features/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    X_train, X_val, X_test, y_train, y_val, y_test, le = prepare_dataset()
    if X_train is None or X_val is None or X_test is None:
        print("Dataset preparation failed. Exiting.")
        return

    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    joblib.dump(le, os.path.join(save_dir, 'label_encoder.joblib'))
    print("Datasets and label encoder saved successfully.")

if __name__ == "__main__":
    main()
