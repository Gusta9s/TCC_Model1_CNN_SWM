import glob, random, shutil

proporcion_test = 0.20
proporcion_validation = 0.10

path_project = '/home/gustavo/Projetos/TCC_Modelo_TensorFlow_SWM'
path_to_lata_src = path_project + '/data/raw/lata'
path_to_pet_src = path_project + '/data/raw/garrafa_pet'
path_to_vinho_src = path_project + '/data/raw/garrafa_de_vinho'
path_to_leite_src = path_project + '/data/raw/garrafa_de_leite'
path_to_lata_dst_validate = path_project + '/data/processed/validation/lata'
path_to_pet_dst_validate = path_project + '/data/processed/validation/garrafa_pet'
path_to_vinho_dst_validate = path_project + '/data/processed/validation/garrafa_de_vinho'
path_to_leite_dst_validate = path_project + '/data/processed/validation/garrafa_de_leite'
path_to_lata_dst_test = path_project + '/data/processed/test/lata'
path_to_pet_dst_test = path_project + '/data/processed/test/garrafa_pet'
path_to_vinho_dst_test = path_project + '/data/processed/test/garrafa_de_vinho'
path_to_leite_dst_test = path_project + '/data/processed/test/garrafa_de_leite'
path_to_lata_dst_train = path_project + '/data/processed/train/lata'
path_to_pet_dst_train = path_project + '/data/processed/train/garrafa_pet'
path_to_vinho_dst_train = path_project + '/data/processed/train/garrafa_de_vinho'
path_to_leite_dst_train = path_project + '/data/processed/train/garrafa_de_leite'

def foreach_lata_files(files_lata):
    for foto in files_lata:
        random_value = random.random()
        filename = foto.split('/')[-1]
        if random_value <= proporcion_validation:
            shutil.move(foto, path_to_lata_dst_validate + '/' + filename)
        elif random_value <= proporcion_validation + proporcion_test:
            shutil.move(foto, path_to_lata_dst_test + '/' + filename)
        else:
            shutil.move(foto, path_to_lata_dst_train + '/' + filename)

def foreach_pet_files(files_pet):
    for foto in files_pet:
        random_value = random.random()
        filename = foto.split('/')[-1]
        if random_value <= proporcion_validation:
            shutil.move(foto, path_to_pet_dst_validate + '/' + filename)
        elif random_value <= proporcion_validation + proporcion_test:
            shutil.move(foto, path_to_pet_dst_test + '/' + filename)
        else:
            shutil.move(foto, path_to_pet_dst_train + '/' + filename)

def foreach_vinho_files(files_vinho):
    for foto in files_vinho:
        random_value = random.random()
        filename = foto.split('/')[-1]
        if random_value <= proporcion_validation:
            shutil.move(foto, path_to_vinho_dst_validate + '/' + filename)
        elif random_value <= proporcion_validation + proporcion_test:
            shutil.move(foto, path_to_vinho_dst_test + '/' + filename)
        else:
            shutil.move(foto, path_to_vinho_dst_train + '/' + filename)

def foreach_leite_files(files_leite):
    for foto in files_leite:
        random_value = random.random()
        filename = foto.split('/')[-1]
        if random_value <= proporcion_validation:
            shutil.move(foto, path_to_leite_dst_validate + '/' + filename)
        elif random_value <= proporcion_validation + proporcion_test:
            shutil.move(foto, path_to_leite_dst_test + '/' + filename)
        else:
            shutil.move(foto, path_to_leite_dst_train + '/' + filename)

def get_files_lata():
    return glob.glob(path_to_lata_src + '/*.jpg')

def get_files_pet():
    return glob.glob(path_to_pet_src + '/*.jpg')

def get_files_vinho():
    return glob.glob(path_to_vinho_src + '/*.jpg')

def get_files_leite():
    return glob.glob(path_to_leite_src + '/*.jpg')

def processo_principal():
    files_lata = get_files_lata()
    files_pet = get_files_pet()
    files_vinho = get_files_vinho()
    files_leite = get_files_leite()

    foreach_lata_files(files_lata)
    foreach_pet_files(files_pet)
    foreach_vinho_files(files_vinho)
    foreach_leite_files(files_leite)

def move_files_to_directories():
    processo_principal()
    print('Processamento concluído com sucesso!')
    print('Arquivos de lata, garrafa pet, garrafa de vinho e garrafa de leite foram movidos para os diretórios de validação, teste e treinamento.')