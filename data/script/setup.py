import glob, random, shutil
from pathlib import Path

proporcion_validation = 0.30

path_to_lata_src = Path('lata')
path_to_garrafa_src = Path('garrafa')
path_to_tenis_src = Path('tenis')
path_to_saco_de_cha_src = Path('sacos_de_cha')

path_to_lata_dst_train = Path('lata_train')
path_to_lata_dst_validate = Path('lata_validation')

path_to_garrafa_dst_train = Path('garrafa_train')
path_to_garrafa_dst_validate = Path('garrafa_validation')

path_to_tenis_dst_train = Path('tenis_train')
path_to_tenis_dst_validate = Path('tenis_validation')

path_to_saco_de_cha_dst_train = Path('sacos_de_cha_train')
path_to_saco_de_cha_dst_validate = Path('sacos_de_cha_validation')

def foreach_garrafa_files(files):
    for foto in files:
        random_value = random.random()
        filename = foto.split('/')[-1]
        if random_value <= proporcion_validation:
            shutil.move(foto, path_to_garrafa_dst_validate / filename)
        else:
            shutil.move(foto, path_to_garrafa_dst_train / filename)

def foreach_lata_files(files):
    for foto in files:
        random_value = random.random()
        filename = foto.split('/')[-1]
        if random_value <= proporcion_validation:
            shutil.move(foto, path_to_lata_dst_validate / filename)
        else:
            shutil.move(foto, path_to_lata_dst_train / filename)

def foreach_tenis_files(files):
    for foto in files:
        random_value = random.random()
        filename = foto.split('/')[-1]
        if random_value <= proporcion_validation:
            shutil.move(foto, path_to_tenis_dst_validate / filename)
        else:
            shutil.move(foto, path_to_tenis_dst_train / filename)

def foreach_sacos_de_cha_files(files):
    for foto in files:
        random_value = random.random()
        filename = foto.split('/')[-1]
        if random_value <= proporcion_validation:
            shutil.move(foto, path_to_saco_de_cha_dst_validate / filename)
        else:
            shutil.move(foto, path_to_saco_de_cha_dst_train / filename)

def get_files_lata():
    return glob.glob(path_to_lata_src + '/*')

def get_files_garrafa():
    return glob.glob(path_to_garrafa_src + '/*')

def get_files_tenis():
    return glob.glob(path_to_tenis_src + '/*')

def get_files_sacos_de_cha():
    return glob.glob(path_to_saco_de_cha_src + '/*')

def processo_principal():
    files_lata = get_files_lata()
    files_garrafa = get_files_garrafa()
    files_tenis = get_files_tenis()
    files_sacos_de_cha = get_files_sacos_de_cha()

    foreach_lata_files(files_lata)
    foreach_garrafa_files(files_garrafa)
    foreach_tenis_files(files_tenis)
    foreach_sacos_de_cha_files(files_sacos_de_cha)

def move_files_to_directories():
    processo_principal()
    print('Processamento concluído com sucesso!')
    print('Os Arquivos foram movidos para os diretórios de validação e treinamento.')