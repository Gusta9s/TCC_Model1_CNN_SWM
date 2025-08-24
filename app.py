from flask import Flask, request, jsonify
import yaml
import logging
import os
from pathlib import Path
from requests import post, RequestException

# Importa as funções dos módulos do seu aplicativo
from src.tcc_modelo_swm.predict import predict_on_image

app = Flask(__name__)

def gerar_imagem_de_rota(origem_latitude, origem_longitude, destino_latitude, destino_longitude):
    """
    Envia uma requisição POST para a API de rotas.
    """
    # O endereço do seu outro serviço
    url_api_rotas = "http://host.docker.internal:3004/api/gerar-imagem-rota"

    # O payload (dados) que a API espera, em formato de dicionário Python
    payload = {
        "origem_latitude": origem_latitude,
        "origem_longitude": origem_longitude,
        "destino_latitude": destino_latitude,
        "destino_longitude": destino_longitude
    }

    # Cabeçalho indicando que estamos enviando dados em formato JSON
    headers = {
        "Content-Type": "application/json"
    }

    print(f"-> Enviando requisição para {url_api_rotas} com payload: {payload}")

    try:
        # Faz a requisição POST.
        response = post(url_api_rotas, json=payload)

        # Verifica se a requisição foi bem-sucedida (código de status 2xx), caso nao lanca excessao com erro e finaliza pilha de execucao.
        response.raise_for_status()
        

        print(f"-> Sucesso! Payload retornado: '{response.json()}'")
        return {"status": "sucesso", "payload": response.json()}

    except RequestException as e:
        # Captura erros de conexão, timeout, status HTTP inválido, etc.
        print(f"ERRO ao contatar a API de rotas: {e}")
        return {"status": "erro", "payload": str(e)}

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
    path_to_yaml = Path('config.yaml')
    try:
        if path_to_yaml.exists():
            config = yaml.safe_load(path_to_yaml.read_text(encoding="utf-8"))
        logging.info("Arquivo de configuração carregado com sucesso.")
        return config
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo de configuração '{path_to_yaml.resolve()}' não foi encontrado.")
        return None

@app.route('/')
def home():
    return "Servidor do Modelo de CNN no ar!"

@app.route('/predict', methods=['POST'])
def predict():
    # Carrega configuração via secret/env
    config = load_config_from_secret()
    if not config:
        return jsonify({"error": "Configuração não encontrada.", "status": "error"}), 500

    setup_logging(config['log_file'])

    # Espera receber {"command": "predict", "image": arquivo binário, "origem_latitude": "origem_latitude", "origem_longitude": "origem_longitude"}
    command = request.form.get('command')
    origem_latitude = request.form.get('origem_latitude')
    origem_longitude = request.form.get('origem_longitude')
    destino_latitude = request.form.get('destino_latitude')
    destino_longitude = request.form.get('destino_longitude')
    image_file = request.files.get('image')

    if not image_file:
        return jsonify({"error": "Arquivo de imagem não fornecido.", "status": "error"}), 400
    if not origem_latitude or not origem_longitude or not destino_latitude or not destino_longitude:
        return jsonify({"erro": "Parâmetros 'origem_latitude', 'origem_longitude', 'destino_latitude' e 'destino_longitude' são obrigatórios e devem ser números."}), 400
    logging.info("--- INICIANDO MODO DE PREDIÇÃO ---")
    predicted_class, confidence = predict_on_image(config, image_file)
    if ((float(confidence) >= 35.0) and (predicted_class != "Vazio")):
        resultado_rota = gerar_imagem_de_rota(origem_latitude, origem_longitude, destino_latitude, destino_longitude)
        if resultado_rota['status'] == 'sucesso':
            payload_da_rota = resultado_rota.get('payload', {})
            
            # 2. Agora sim, verifique o conteúdo do payload com segurança
            if payload_da_rota.get('success') is True and payload_da_rota.get('filename') is not None:
                logging.info(f"Rota gerada com sucesso. Arquivo: {payload_da_rota['filename']}")
                return jsonify({
                    "prediction": predicted_class, 
                    "confidence": float(confidence), 
                    "status": "success"
                })
            else:
                # A API de rotas funcionou, mas reportou uma falha interna
                logging.warning(f"Falha ao gerar rota (resposta da API): {payload_da_rota}")
                return jsonify({
                    "prediction": predicted_class, 
                    "confidence": float(confidence), 
                    "status": "warning",
                    "route_error": "API de rotas reportou falha na geração do arquivo."
                })
        
        else: # Se resultado_rota['status'] == 'erro'
            # A requisição para a API de rotas falhou (conexão, etc.)
            logging.error(f"Não foi possível contatar a API de rotas: {resultado_rota['payload']}")
            return jsonify({
                "prediction": predicted_class, 
                "confidence": float(confidence), 
                "status": "warning",
                "route_error": "Não foi possível conectar ao serviço de rotas."
            })
    else:
        logging.warning(f"Threshold não atingido: {predicted_class}, {confidence}")
        return jsonify({
            "prediction": predicted_class, 
            "confidence": float(confidence), 
            "status": "warning"
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)