from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow import expand_dims
from tensorflow.nn import softmax
from numpy import argmax, max
import logging
from io import BytesIO

def predict_on_image(config: dict, image_file):
    """
    Carrega um modelo treinado e faz uma previsão em uma única imagem recebida como arquivo.
    
    Args:
        config (dict): Dicionário de configuração.
        image_file: Arquivo de imagem recebido (objeto FileStorage do Flask).
    
    Returns:
        tuple: (nome_da_classe_prevista, confiança)
    """

    try:
        logging.info(f"Carregando modelo de: {config['prediction']['model_path']}")
        model = load_model(config['prediction']['model_path'])
        
        image_size = tuple(config['data']['image_size'])
        class_names = config['prediction']['class_names']

        img = load_img(BytesIO(image_file.read()), target_size=image_size)
        img_array = img_to_array(img)
        img_array = expand_dims(img_array, 0) # Cria um batch

        predictions = model.predict(img_array)
        score = softmax(predictions[0])
        
        predicted_class = class_names[argmax(score)]
        confidence = 100 * max(score)

        logging.info(
            f"A imagem enviada pertence à classe '{predicted_class}' com {confidence:.2f}% de confiança."
        )

        print(f"A imagem enviada pertence à classe '{predicted_class}' com {confidence:.2f}% de confiança.")
        
        return predicted_class, confidence

    except Exception as e:
        logging.error(f"Ocorreu um erro durante a predição: {e}")
        return None, None