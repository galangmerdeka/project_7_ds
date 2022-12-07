from fastapi import APIRouter
import data_models as dm
from api.v1 import backend

router = APIRouter(
    prefix="/api/v2",
    tags=["v2"]
)

@router.post("/predict")
async def predict(data: dm.Applicant):
    prediction = backend.predict(data)
    return {
        "result": prediction
    }