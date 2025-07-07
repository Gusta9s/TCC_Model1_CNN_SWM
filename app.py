from flask import Flask, request, jsonify
import tensorflow as tf
import yaml
import logging
import os

# Importa as funções dos módulos do seu aplicativo
from src.tcc_modelo_swm.data_loader import load_datasets
from src.tcc_modelo_swm.model import create_model
from src.tcc_modelo_swm.train import train_model
from src.tcc_modelo_swm.predict import predict_on_image
from data.script.setup import move_files_to_directories

app = Flask(__name__)

def setup_logging(log_file_path: str):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler()
        ]
    )

def load_config_from_secret():
    config_path = os.environ.get('CONFIG_SECRET_PATH', 'config.yaml')
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Arquivo de configuração carregado com sucesso.")
        return config
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo de configuração '{config_path}' não foi encontrado.")
        return None

@app.route('/')
def home():
    return "Servidor do Modelo de CNN no ar!"

@app.route('/predict', methods=['POST'])
def predict():
    # Carrega configuração via secret/env
    config = load_config_from_secret()
    if not config:
        return jsonify({"error": "Configuração não encontrada."}), 500

    setup_logging(config['log_file'])

    # Espera receber {"command": "train"} ou {"command": "predict", "image": arquivo binário}
    if request.content_type.startswith('application/json'):
        data = request.get_json(force=True)
        command = data.get('command')
        image_file = None
    else:
        command = request.form.get('command')
        image_file = request.files.get('image')

    if command == 'train':
        logging.info("--- INICIANDO PIPELINE DE TREINAMENTO ---")
        move_files_to_directories()
        train_ds, val_ds = load_datasets(config)
        if train_ds is None:
            logging.error("Finalizando execução devido a erro no carregamento de dados.")
            return jsonify({"error": "Erro no carregamento dos dados."}), 500
        model = create_model(config)
        model.summary(print_fn=logging.info)
        train_model(config, model, train_ds, val_ds)
        logging.info("--- PIPELINE DE TREINAMENTO CONCLUÍDO ---")
        return jsonify({"message": "Treinamento concluído com sucesso."})

    elif command == 'predict':
        if not image_file:
            return jsonify({"error": "Arquivo de imagem não fornecido."}), 400
        logging.info("--- INICIANDO MODO DE PREDIÇÃO ---")
        predicted_class, confidence = predict_on_image(config, image_file)
        logging.info("--- PREDIÇÃO CONCLUÍDA ---")
        return jsonify({"prediction": predicted_class, "confidence": confidence})

    else:
        return jsonify({"error": "Comando inválido. Use 'train' ou 'predict'."}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)