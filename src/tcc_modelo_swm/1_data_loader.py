from tensorflow.keras.utils import image_dataset_from_directory
import logging

train_dataset_from_lata = image_dataset_from_directory(lata_folder_train,
                                             image_size=(),
                                             batch_size=32)


train_dataset_from_pet = image_dataset_from_directory(pet_folder_train,
                                             image_size=(256, 256),
                                             batch_size=32)

train_dataset_from_vinho = image_dataset_from_directory(vinho_folder_train,
                                             image_size=(256, 256),
                                             batch_size=32)

train_dataset_from_leite = image_dataset_from_directory(leite_folder_train,
                                             image_size=(256, 256),
                                             batch_size=32)

validation_dataset_from_lata = image_dataset_from_directory(lata_folder_validation,
                                                  image_size=(256, 256),
                                                  batch_size=32)

validation_dataset_from_pet = image_dataset_from_directory(pet_folder_validation,
                                                  image_size=(256, 256),
                                                  batch_size=32)

validation_dataset_from_vinho = image_dataset_from_directory(vinho_folder_validation,
                                                  image_size=(256, 256),
                                                  batch_size=32)

validation_dataset_from_leite = image_dataset_from_directory(leite_folder_validation,
                                                  image_size=(256, 256),
                                                  batch_size=32)

test_dataset_from_lata = image_dataset_from_directory(lata_folder_test,
                                            image_size=(256, 256),
                                            batch_size=32)

test_dataset_from_pet = image_dataset_from_directory(pet_folder_test,
                                            image_size=(256, 256),
                                            batch_size=32)

test_dataset_from_vinho = image_dataset_from_directory(vinho_folder_test,
                                            image_size=(256, 256),
                                            batch_size=32)

test_dataset_from_leite = image_dataset_from_directory(leite_folder_test,
                                            image_size=(256, 256),
                                            batch_size=32)

def load_datasets(config: dict):
    """
    Carrega os datasets de treinamento, validação e teste a partir dos diretórios.
    
    Args:
        config (dict): Dicionário de configuração carregado do config.yaml.
        
    Returns:
        tuple: Contendo os datasets (train_ds, val_ds, test_ds).
    """

    try:
        lata_path_train = config['data']['lata_folder_train']
        pet_path_train = config['data']['pet_folder_train']
        vinho_path_train = config['data']['vinho_folder_train']
        leite_path_train = config['data']['leite_folder_train']
        lata_path_test = config['data']['lata_folder_test']
        pet_path_test = config['data']['pet_folder_test']
        vinho_path_test = config['data']['vinho_folder_test']
        leite_path_test = config['data']['leite_folder_test']
        lata_path_validation = config['data']['lata_folder_validation']
        pet_path_validation = config['data']['pet_folder_validation']
        vinho_path_validation = config['data']['vinho_folder_validation']
        leite_path_validation = config['data']['leite_folder_validation']
        image_size = tuple(config['data']['image_size'])
        batch_size = config['data']['batch_size']

        logging.info(f"Carregando dados de: {base_path}")
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
            f"{base_path}/train",
            labels='inferred',
            label_mode='int',
            image_size=image_size,
            interpolation='nearest', # Usar 'nearest' para evitar artefatos em algumas imagens
            batch_size=batch_size,
            shuffle=True
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            f"{base_path}/validation",
            labels='inferred',
            label_mode='int',
            image_size=image_size,
            interpolation='nearest',
            batch_size=batch_size,
            shuffle=False
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            f"{base_path}/test",
            labels='inferred',
            label_mode='int',
            image_size=image_size,
            interpolation='nearest',
            batch_size=batch_size,
            shuffle=False
        )
        
        logging.info("Datasets de treinamento, validação e teste carregados com sucesso.")
        return train_ds, val_ds, test_ds

    except FileNotFoundError as e:
        logging.error(f"Erro: Diretório de dados não encontrado - {e}")
        return None, None, None
