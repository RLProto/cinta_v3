from pydantic import BaseModel


class InferenceModel(BaseModel):
    prediction: str
    accuracy: float
    

class InferenceResponseModel(InferenceModel):
    image_name: str