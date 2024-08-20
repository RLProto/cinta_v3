from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.model.inference_response_model import InferenceResponseModel
from app.services.inference_model_cloud import InferenceModelCloud
from app.services.manager_model import ManagerModel

router = APIRouter(tags=["Manager Model Cloud"])

manager_model_instance = ManagerModel()
model_inference = None

def get_manager_model() -> ManagerModel:
    return manager_model_instance

def get_inference_model_cloud() -> InferenceModelCloud:
    return model_inference

@router.get("/models", response_model=dict)
def list_models() -> dict:
    return {"models": ["model1", "model2"]}

@router.post("/upload-model")
async def upload_model(file: UploadFile = File(...), manager_model: ManagerModel = Depends(get_manager_model)) -> dict:
    global model_inference
    if await manager_model.load_model(file):
        manager_model.save_model()
        model_inference = InferenceModelCloud(manager_model.get_model(), manager_model.get_classes())
        return {"filename": file.filename}
    raise HTTPException(status_code=415, detail=f"Arquivo '{file.filename}' tipo inválido. Modelo não treinado no Soda Vision.")

@router.post("/inference", response_model=InferenceResponseModel)
async def run_inference(file: UploadFile = File(...),
                        model_inference: InferenceModelCloud = Depends(get_inference_model_cloud)) -> InferenceResponseModel:
    try:
        response : InferenceResponseModel = await model_inference.predict(file)
    except:
        raise HTTPException(status_code=415, detail=f"Modelo Soda vision Clound não enviado!")
    return response
