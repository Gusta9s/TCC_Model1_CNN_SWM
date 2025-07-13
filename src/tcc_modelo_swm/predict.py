from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow import expand_dims
from tensorflow.nn import softmax
from numpy import argmax, max
import logging
from io import BytesIO
from pathlib import Path

def predict_on_image(config: dict, image_file, train_ds):
    """
    Carrega um modelo treinado e faz uma previsão em uma única imagem recebida como arquivo.
    
    Args:
        config (dict): Dicionário de configuração.
        image_file: Arquivo de imagem recebido (objeto FileStorage do Flask).
    
    Returns:
        tuple: (nome_da_classe_prevista, confiança)
    """

    try:
        path_to_model = Path('./models/modelo.keras')
        logging.info(f"Carregando modelo de: {path_to_model.resolve()}")
        model = load_model(path_to_model.resolve())
        
        image_size = tuple(config['data']['image_size'])
        class_names = train_ds.class_names if train_ds else config['prediction']['class_names']

        # Define o limiar de confiança mínimo (ex: 75%)
        THRESHOLD = 35.0

        img = load_img(BytesIO(image_file.read()), target_size=image_size)
        img_array = img_to_array(img)
        img_array = expand_dims(img_array, 0) # Cria um batch

        predictions = model.predict(img_array)
        score = softmax(predictions[0])
        
        predicted_class = class_names[argmax(score)]
        confidence = 100 * max(score)

        if confidence < THRESHOLD:
            logging.info(
                f"Previsão descartada. Confiança de {confidence:.2f}% é inferior ao limiar de {THRESHOLD}%."
            )

            return "Nenhum objeto detectado", confidence

        logging.info(
            f"A imagem enviada pertence à classe '{predicted_class}' com {confidence:.2f}% de confiança."
        )

        print(f"A imagem enviada pertence à classe '{predicted_class}' com {confidence:.2f}% de confiança.")
        
        return predicted_class, confidence

    except Exception as e:
        logging.error(f"Ocorreu um erro durante a predição: {e}")
        return None, None