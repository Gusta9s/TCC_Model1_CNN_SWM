import random, shutil
from pathlib import Path

proporcion_validation = 0.30

path_to_lata_src = Path('./data/raw/lata')
path_to_garrafa_src = Path('./data/raw/garrafa')
path_to_tenis_src = Path('./data/raw/tenis')
path_to_caixa_de_papelao_src = Path('./data/raw/caixa_de_papelao')

path_to_lata_dst_train = Path('./data/processed/train/lata')
path_to_lata_dst_validate = Path('./data/processed/validation/lata')

path_to_garrafa_dst_train = Path('./data/processed/train/garrafa')
path_to_garrafa_dst_validate = Path('./data/processed/validation/garrafa')

path_to_tenis_dst_train = Path('./data/processed/train/tenis')
path_to_tenis_dst_validate = Path('./data/processed/validation/tenis')

path_to_caixa_de_papelao_dst_train = Path('./data/processed/train/caixa_de_papelao')
path_to_caixa_de_papelao_dst_validate = Path('./data/processed/validation/caixa_de_papelao')

def foreach_garrafa_files(files):
    for foto in files:
        random_value = random.random()
        if random_value <= proporcion_validation:
            shutil.move(foto.resolve(), path_to_garrafa_dst_validate.resolve() / foto.name)
        else:
            shutil.move(foto.resolve(), path_to_garrafa_dst_train.resolve() / foto.name)

def foreach_lata_files(files):
    for foto in files:
        random_value = random.random()
        if random_value <= proporcion_validation:
            shutil.move(foto.resolve(), path_to_lata_dst_validate.resolve() / foto.name)
        else:
            shutil.move(foto.resolve(), path_to_lata_dst_train.resolve() / foto.name)

def foreach_tenis_files(files):
    for foto in files:
        random_value = random.random()
        if random_value <= proporcion_validation:
            shutil.move(foto.resolve(), path_to_tenis_dst_validate.resolve() / foto.name)
        else:
            shutil.move(foto.resolve(), path_to_tenis_dst_train.resolve() / foto.name)

def foreach_caixa_de_papelao_files(files):
    for foto in files:
        random_value = random.random()
        if random_value <= proporcion_validation:
            shutil.move(foto.resolve(), path_to_caixa_de_papelao_dst_validate.resolve() / foto.name)
        else:
            shutil.move(foto.resolve(), path_to_caixa_de_papelao_dst_train.resolve() / foto.name)

def get_files_lata():
    return path_to_lata_src.glob('*')

def get_files_garrafa():
    return path_to_garrafa_src.glob('*')

def get_files_tenis():
    return path_to_tenis_src.glob('*')

def get_files_caixa_de_papelao():
    return path_to_caixa_de_papelao_src.glob('*')

def processo_principal():
    files_lata = get_files_lata()
    files_garrafa = get_files_garrafa()
    files_tenis = get_files_tenis()
    files_caixa_de_papelao = get_files_caixa_de_papelao()

    foreach_lata_files(files_lata)
    foreach_garrafa_files(files_garrafa)
    foreach_tenis_files(files_tenis)
    foreach_caixa_de_papelao_files(files_caixa_de_papelao)

def move_files_to_directories():
    processo_principal()
    print('Processamento concluído com sucesso!')
    print('Os Arquivos foram movidos para os diretórios de validação e treinamento.')