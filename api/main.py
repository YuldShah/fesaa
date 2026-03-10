from fastapi import Depends, FastAPI

from api.dependencies import verify_api_key
from api.routers.assessments import router as assessments_router
from core.settings import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name)
app.include_router(
    assessments_router,
    prefix="/v1/assessments",
    tags=["assessments"],
    dependencies=[Depends(verify_api_key)],
)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
