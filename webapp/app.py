import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from predict import predict_image_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_WEIGHTS = PROJECT_ROOT / "lung_cancer_model.pth"

app = FastAPI(title="CT Scan Report Generator")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/report")
async def create_report(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        return {
            "ok": False,
            "error": "Unsupported file type. Please upload an image (png/jpg/jpeg/bmp/tif/tiff).",
        }

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

    try:
        report = predict_image_report(tmp_path, weights_path=str(DEFAULT_WEIGHTS))
        return {"ok": True, "report": report}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

