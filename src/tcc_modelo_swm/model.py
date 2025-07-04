import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling, Input
import logging


def create_model(config: dict) -> tf.keras.Model:
    """
    Cria e retorna o modelo CNN baseado na configuração.
    
    Args:
        config (dict): Dicionário de configuração.
        
    Returns:
        tf.keras.Model: O modelo Keras compilado.
    """

    input_shape = tuple(config['model']['input_shape'])
    num_classes = config['model']['num_classes']
    
    logging.info("Criando a arquitetura do modelo CNN...")
    
    model = Sequential([
        Input(shape=input_shape), # Define a forma de entrada do modelo
        Rescaling(1./255), # Tranforma a imagem para tons de cinza por filtragem

        # Bloco Convolucional 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # Bloco Convolucional 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # Bloco Convolucional 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # Camadas de Classificação
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Dropout para reduzir overfitting
        Dense(num_classes, activation='softmax') # Camada de saída
    ])

    logging.info("Modelo criado com sucesso.")
    return model