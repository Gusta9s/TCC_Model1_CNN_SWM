from tensorflow.keras.utils import image_dataset_from_directory

project_folder = '/home/gustavo/Projetos/TCC_Modelo_TensorFlow_SWM/'

lata_folder_validation = project_folder + 'data/processed/validation/lata'
pet_folder_validation = project_folder + 'data/processed/validation/garrafa_pet'
vinho_folder_validation = project_folder + 'data/processed/validation/garrafa_de_vinho'
leite_folder_validation = project_folder + 'data/processed/validation/garrafa_de_leite'
lata_folder_test = project_folder + 'data/processed/test/lata'
pet_folder_test = project_folder + 'data/processed/test/garrafa_pet'
vinho_folder_test = project_folder + 'data/processed/test/garrafa_de_vinho'
leite_folder_test = project_folder + 'data/processed/test/garrafa_de_leite'
lata_folder_train = project_folder + 'data/processed/train/lata'
pet_folder_train = project_folder + 'data/processed/train/garrafa_pet'
vinho_folder_train = project_folder + 'data/processed/train/garrafa_de_vinho'
leite_folder_train = project_folder + 'data/processed/train/garrafa_de_leite'

train_dataset_from_lata = image_dataset_from_directory(lata_folder_train,
                                             image_size=(256, 256),
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
