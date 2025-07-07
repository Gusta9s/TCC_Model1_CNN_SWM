import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import logging

def train_model(config: dict, model: tf.keras.Model, train_ds, val_ds):
    """
    Compila e treina o modelo.
    
    Args:
        config (dict): Dicionário de configuração.
        model (tf.keras.Model): O modelo a ser treinado.
        train_ds (tf.data.Dataset): Dataset de treinamento.
        val_ds (tf.data.Dataset): Dataset de validação.
    """

    logging.info("Iniciando o processo de treinamento.")
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath=config['training']['save_model_path'],
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5, # Número de épocas sem melhora antes de parar
        restore_best_weights=True,
        verbose=1
    )

    # Compilação
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
        loss=config['training']['loss_function'],
        metrics=config['training']['metrics']
    )
    
    logging.info(f"Modelo compilado com otimizador={config['training']['optimizer']} e loss={config['training']['loss_function']}")
    
    model.fit(
        train_ds,
        epochs=config['training']['epochs'],
        validation_data=val_ds,
        callbacks=[checkpoint, early_stopping]
    )
    logging.info(f"Treinamento concluído. Melhor modelo salvo em: {config['training']['save_model_path']}")