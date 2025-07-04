import yaml
import argparse
import logging
import os

# Importa as funções dos módulos do seu aplicativo
from src.tcc_modelo_swm.data_loader import load_datasets
from src.tcc_modelo_swm.model import create_model
from src.tcc_modelo_swm.train import train_model
from src.tcc_modelo_swm.predict import predict_on_image
import src.tcc_modelo_swm.api
import data.script.setup


def setup_logging(log_file_path: str):
    """Configura o sistema de logging para salvar em arquivo e exibir no console."""
    # Garante que o diretório de logs exista
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Remove handlers existentes
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'), # Salva no arquivo
            logging.StreamHandler() # Exibe no console
        ]
    )

def load_config(config_path: str) -> dict:
    """Carrega o arquivo de configuração YAML."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Arquivo de configuração carregado com sucesso.")
        return config
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo de configuração '{config_path}' não foi encontrado.")
        return None

def main():
    # --- Configuração do Parser de Argumentos ---
    parser = argparse.ArgumentParser(description="Orquestrador para o projeto de CNN.")
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml', 
        help='Caminho para o arquivo de configuração YAML.'
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Comandos disponíveis: train, predict')

    # Sub-parser para o comando 'train'
    train_parser = subparsers.add_parser('train', help='Executa o pipeline de treinamento completo.')
    
    # Sub-parser para o comando 'predict'
    predict_parser = subparsers.add_parser('predict', help='Executa uma predição em uma imagem.')
    predict_parser.add_argument(
        '--image', 
        type=str, 
        required=True, 
        help='Caminho para a imagem que será classificada.'
    )

    args = parser.parse_args()
    
    # --- Carregar Configuração e Iniciar Logging ---
    config = load_config(args.config)
    if not config:
        return # Encerra se a configuração não puder ser carregada
        
    setup_logging(config['log_file'])

    # --- Execução do Comando Escolhido ---
    if args.command == 'train':
        logging.info("--- INICIANDO PIPELINE DE TREINAMENTO ---")

        data.script.setup.move_files_to_directories()

        # 1. Carregar Dados
        train_ds, val_ds, test_ds = load_datasets(config)
        if train_ds is None:
            logging.error("Finalizando execução devido a erro no carregamento de dados.")
            return
            
        # 2. Criar Modelo
        model = create_model(config)
        model.summary(print_fn=logging.info) # Envia o summary para o log
        
        # 3. Treinar Modelo
        train_model(config, model, train_ds, val_ds)
        logging.info("--- PIPELINE DE TREINAMENTO CONCLUÍDO ---")

    elif args.command == 'predict':
        logging.info("--- INICIANDO MODO DE PREDIÇÃO ---")
        predict_on_image(config, args.image)
        logging.info("--- PREDIÇÃO CONCLUÍDA ---")

if __name__ == '__main__':
    main()