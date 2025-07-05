import tensorflow as tf
import os
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import LSTM, Dropout, Dense #type:ignore
from tensorflow.keras.utils import plot_model #type:ignore

def construir_modelo_LSTM(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


modelo = construir_modelo_LSTM((30, 100), 6)
plot_model(modelo, to_file='modelo_lstm.png', show_shapes=True, show_layer_names=True)