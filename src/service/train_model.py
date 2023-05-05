import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from src.service.balance_df import generate_balaced_dataset
from src.service.df_to_dataset import df_to_dataset


embedding_gb = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3" 
hub_layer_gb = hub.KerasLayer(embedding_gb, dtype=tf.string, trainable=True)
    
def train_model():
    global df_balanced

    balanced_file = os.listdir('src/dataset/balanced')
    if len(balanced_file) == 0:
        print('No existe un dataset balanceado, se generará uno')
        df_balanced = generate_balaced_dataset()
    else:
        print('Existe un dataset balanceado, leyendo...')
        df_balanced = pd.read_csv('src/dataset/balanced/balanced_out.csv')

    train, val, test = np.split(df_balanced.sample(frac=1), [int(0.8*len(df_balanced)), int(0.9*len(df_balanced))])

    print("Generando datasets...")
    train_data = df_to_dataset(train)
    valid_data = df_to_dataset(val)
    test_data = df_to_dataset(test)

    embedding = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3" 
    hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)
    
    print("Generando modelo...")

    modelTF = tf.keras.Sequential()
    modelTF.add(hub_layer)
    modelTF.add(tf.keras.layers.Dense(16, activation='relu'))
    modelTF.add(tf.keras.layers.Dropout(0.4))
    modelTF.add(tf.keras.layers.Dense(16, activation='relu'))
    modelTF.add(tf.keras.layers.Dropout(0.4))
    modelTF.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    modelTF.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
    
    print("Entrenando modelos...")
    print("Modelo de entrenamiento: ")
    modelTF.evaluate(train_data)
    print("Modelo de validación: ")
    modelTF.evaluate(valid_data)
    print("Modelo de prueba: ")
    modelTF.evaluate(test_data)
    
    modelTF.fit(train_data, epochs=5, validation_data=valid_data)

    print("Guardando modelo...")
    # save to hub_model 
    
    modelTF.save('src/models/model.h5')
def use_model():
    global model
    custom_objects={'KerasLayer': hub_layer_gb}


    try:
        model = tf.keras.models.load_model('src/models/model.h5', custom_objects=custom_objects)
    except:
        print("No existe un modelo entrenado, se entrenará uno")
        train_model()
        model = tf.keras.models.load_model('src/models/model.h5', custom_objects=custom_objects)
    
    return model

