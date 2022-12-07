from fastapi import APIRouter
import data_models as dm
from api.v1 import backend

router = APIRouter(
    prefix="/api/v1",
    tags=["v1"]
)

@router.post("/predict")
async def predict(data: dm.Applicant):
    prediction = backend.predict(data)
    return {
        "result": prediction
    }