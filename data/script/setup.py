import glob, random, shutil

proporcion_validation = 0.30

path_project = '/home/gustavo/Projetos/TCC_Modelo_TensorFlow_SWM'
path_to_lata_src = path_project + '/data/raw/lata'
path_to_garrafa_src = path_project + '/data/raw/garrafa'
path_to_tenis_src = path_project + '/data/raw/tenis'
path_to_saco_de_cha_src = path_project + '/data/raw/sacos_de_cha'

path_to_lata_dst_train = path_project + '/data/processed/train/lata'
path_to_lata_dst_validate = path_project + '/data/processed/validation/lata'

path_to_garrafa_dst_train = path_project + '/data/processed/train/garrafa'
path_to_garrafa_dst_validate = path_project + '/data/processed/validation/garrafa'

path_to_tenis_dst_train = path_project + '/data/processed/train/tenis'
path_to_tenis_dst_validate = path_project + '/data/processed/validation/tenis'

path_to_saco_de_cha_dst_train = path_project + '/data/processed/train/sacos_de_cha'
path_to_saco_de_cha_dst_validate = path_project + '/data/processed/validation/sacos_de_cha'

def foreach_garrafa_files(files):
    for foto in files:
        random_value = random.random()
        filename = foto.split('/')[-1]
        if random_value <= proporcion_validation:
            shutil.move(foto, path_to_garrafa_dst_validate + '/' + filename)
        else:
            shutil.move(foto, path_to_garrafa_dst_train + '/' + filename)

def foreach_lata_files(files):
    for foto in files:
        random_value = random.random()
        filename = foto.split('/')[-1]
        if random_value <= proporcion_validation:
            shutil.move(foto, path_to_lata_dst_validate + '/' + filename)
        else:
            shutil.move(foto, path_to_lata_dst_train + '/' + filename)

def foreach_tenis_files(files):
    for foto in files:
        random_value = random.random()
        filename = foto.split('/')[-1]
        if random_value <= proporcion_validation:
            shutil.move(foto, path_to_tenis_dst_validate + '/' + filename)
        else:
            shutil.move(foto, path_to_tenis_dst_train + '/' + filename)

def foreach_sacos_de_cha_files(files):
    for foto in files:
        random_value = random.random()
        filename = foto.split('/')[-1]
        if random_value <= proporcion_validation:
            shutil.move(foto, path_to_saco_de_cha_dst_validate + '/' + filename)
        else:
            shutil.move(foto, path_to_saco_de_cha_dst_train + '/' + filename)

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