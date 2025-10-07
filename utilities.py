import os, json
from itertools import chain, combinations
import numpy as np
import tensorflow as tf
from constants import (
    temp_data_folder_name, models_folder,
    save_evaluation_model_results_file_name,
    save_the_best_combination_file_name,
    save_combinations_file_name,
    feature_name_mapping, feature_name_reverse_mapping
)

def create_temp_data_folder():
    os.makedirs(temp_data_folder_name, exist_ok=True)
    return temp_data_folder_name

def encode_combination(combination):
    # dacă e o listă de nume (A..O în constants), codează; dacă sunt indexuri, întoarce string-ul original
    try:
        return "".join(feature_name_mapping[f] for f in combination)
    except Exception:
        return ",".join(map(str, combination))

def decode_combination(encoded_combination):
    if all(c in feature_name_reverse_mapping for c in encoded_combination):
        return [feature_name_reverse_mapping[c] for c in encoded_combination]
    return [int(x) if x.isdigit() else x for x in encoded_combination.split(",")]

def save_best_combination(file_name=save_combinations_file_name, n=10):
    path = os.path.join(temp_data_folder_name, file_name)
    combos=[]
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(0, len(lines), 2):
            acc_line = lines[i].strip()
            cmb_line = lines[i+1].strip()
            acc = float(acc_line.split(": ")[1].strip("%"))/100.0
            cmb = cmb_line.split(": ")[1]
            combos.append((acc, cmb))
    except FileNotFoundError:
        print(f"File not found: {path}")
        return
    combos.sort(key=lambda x: x[0], reverse=True)
    os.makedirs(temp_data_folder_name, exist_ok=True)
    out = os.path.join(temp_data_folder_name, save_the_best_combination_file_name)
    with open(out, "w") as g:
        for i, (acc, cmb) in enumerate(combos[:n], start=1):
            g.write(f"{i}. Accuracy: {acc:.2%}, Combination: {cmb}\n")
    print(f"[OK] Top {n} combinații salvate în {out}")

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units=256, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        d = input_shape[-1]
        self.W1 = self.add_weight(shape=(d, self.units), initializer='glorot_uniform', name="W1")
        self.b1 = self.add_weight(shape=(self.units,), initializer='zeros', name="b1")
        self.W2 = self.add_weight(shape=(self.units, 1), initializer='glorot_uniform', name="W2")
        self.b2 = self.add_weight(shape=(1,), initializer='zeros', name="b2")
        super().build(input_shape)
    def call(self, x, return_attention=False):
        q = tf.matmul(x, self.W1) + self.b1
        q = tf.nn.tanh(q)
        e = tf.matmul(q, self.W2) + self.b2
        a = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(x * a, axis=1)
        if return_attention:
            return context, a
        return context
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg
