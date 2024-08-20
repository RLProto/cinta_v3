from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image

from app.model.inference_response_model import InferenceResponseModel
from app.services.validator.model_validator import ModelValidator


class InferenceModelCloud:
    
    def __init__(self, model_h5, classes):
        self.classes = classes
        self.model_h5 = model_h5

    def check_model_binary(self):
        if len(self.classes) <= 2:
            return True
        return False
            
    async def predict(self, image_file) -> InferenceResponseModel:
        ModelValidator.validate_image_data(image_file)
        ModelValidator.validate_model_loaded(image_file)
        image_bytes = await image_file.read()
        image = BytesIO(image_bytes)
        image = Image.open(image)
        target_size = (224, 224)
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        predition = self.model_h5.predict(image_array)
        if self.check_model_binary():
            if predition[0] > 0.5:
                predict_class = self.classes[1]
            else:
                predict_class = self.classes[0]
            prediction = {
                "type": "Classificacao Binaria",
                "prediction": predict_class,
                "accuracy": predition[0],
                "image_name": image_file.filename
            }
        else:
            class_index = np.argmax(predition)
            accuracy = predition[0][class_index]
            predict_class = self.classes[class_index]
            prediction = {
                "type": "Classificacao Multiclasse",
                "prediction": predict_class,
                "accuracy": accuracy,
                "image_name": image_file.filename
            }
        return InferenceResponseModel(**prediction)