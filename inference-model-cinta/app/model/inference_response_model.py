from typing import Optional

from pydantic import BaseModel


class InferenceModel(BaseModel):
    type: str
    prediction: str
    accuracy: Optional[float]
    

class InferenceResponseModel(InferenceModel):
    image_name: str