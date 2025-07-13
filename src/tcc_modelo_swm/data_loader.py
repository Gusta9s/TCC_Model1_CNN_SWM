import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory as ims
from pathlib import Path
import logging


def load_datasets(config: dict):
    """
    Carrega os datasets de treinamento e validação a partir dos diretórios.
    
    Args:
        config (dict): Dicionário de configuração carregado do config.yaml.
        
    Returns:
        tuple: Contendo os datasets (train_ds, val_ds).
    """

    try:
        train_ds = Path('./data/processed/train')
        val_ds = Path('./data/processed/validation')
        image_size = tuple(config['data']['image_size'])
        batch_size = config['data']['batch_size']

        logging.info(f"Carregando dados de: {train_ds.resolve()}")

        train_ds_completo = ims(
            train_ds.resolve(),
            labels='inferred', # Infire as classes a partir dos nomes das pastas
            label_mode='int', # 0 para 'caixa_de_detergente', 1 para 'caixa_de_leite', etc.
            image_size=image_size,
            interpolation='nearest',
            batch_size=batch_size,
            shuffle=True
        )

        logging.info(f"Carregando dados de: {val_ds.resolve()}")

        val_ds_completo = ims(
            val_ds.resolve(),
            labels='inferred',
            label_mode='int',
            image_size=image_size,
            interpolation='nearest',
            batch_size=batch_size,
            shuffle=False
        )
    
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds_completo = train_ds_completo.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds_completo = val_ds_completo.cache().prefetch(buffer_size=AUTOTUNE)

        logging.info("Datasets de treinamento e validação carregados com sucesso.")
        return train_ds_completo, val_ds_completo

    except FileNotFoundError as e:
        logging.error(f"Erro: Diretório de dados não encontrado - {e}")
        return None, None