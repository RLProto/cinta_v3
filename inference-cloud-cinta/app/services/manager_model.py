import io
import os
import zipfile

import tensorflow as tf


class ManagerModel:
    def __init__(self):
        self.model_tflite_file_name = "model_validation.h5"
        self.path_model_h5 = "models/validation/model_validation.h5"
        self.classes_path = "classes.txt"
        self.EXTRACT_FILES = [self.path_model_h5, self.classes_path]
        self.extracted_files = {}
        print(os.path.abspath(os.getcwd()))
        self.model_path = os.path.join("/workspace", "app", "data", self.model_tflite_file_name)
        print(self.model_path)

    async def load_model(self, file):
        contents = await file.read()
        file_like_object = io.BytesIO(contents)
        with zipfile.ZipFile(file_like_object, "r") as zip_ref:
            for filename in zip_ref.namelist():
                if filename in self.EXTRACT_FILES:
                    extracted_file = zip_ref.read(filename)
                    self.extracted_files[filename] = extracted_file
                    print(f"Arquivo {filename} extraído")
        return self.path_model_h5 in self.extracted_files
    
    def save_model(self):
        model_bytes = self.extracted_files[self.path_model_h5]
        with open(self.model_path, "wb") as model_file:
            model_file.write(model_bytes)
        print(f"Modelo salvo em {self.model_path}")
    
    def get_model(self):
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 0:
            self.model = self.get_format_h5_model(self.model_path)
            print(f"Modelo carregado de {self.model_path}")
            return self.model
        else:
            print(f"Erro: Arquivo do modelo não existe ou está vazio.")
            return None
        
    def get_classes(self):
        classes = self.extracted_files.get(self.classes_path, None)
        return classes.decode("utf-8").splitlines()

    def get_format_h5_model(self, model_path):
        model = tf.keras.models.load_model(model_path)
        return model