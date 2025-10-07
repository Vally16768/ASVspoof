import os

# === Locația datasetului (ASVspoof 2019 LA) ===
# Dacă ASVSPOOF_ROOT nu e setat, cade pe "dataset"
directory = os.getenv("ASVSPOOF_ROOT", "dataset")

# === Setări dataset/split ===
sampling_rate   = 16000
random_state    = 42
test_size       = 0.20     # pentru split train/test (dacă e folosit)
validation_size = 0.20     # procent din train -> val (dacă e folosit)

# === Foldere/fișiere și denumiri ===
results_folder                          = "results"
temp_data_folder_name                   = "temp_data"
models_folder                           = "models"
save_evaluation_model_results_file_name = "eval_results.json"
save_the_best_combination_file_name     = "combinations_ordered_by_accuracy.txt"
save_combinations_file_name             = "combinations_accuracy.txt"

# === Mapări pentru denumiri de trăsături ===
feature_name_mapping = {
    "mfcc": "A",
    "mfcc_delta": "B",
    "centroid": "C",
    "bandwidth": "D",
    "contrast": "E",
    "rolloff": "F",
    "flatness": "G",
    "chroma": "H",
    "rms": "I",
    "zcr": "J",
}
feature_name_reverse_mapping = {v: k for k, v in feature_name_mapping.items()}
