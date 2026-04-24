from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from paddleocr import PaddleOCR

app = FastAPI()

# Global OCR object (loaded once)
ocr = None


# Load model once at startup (IMPORTANT for Render)
@app.on_event("startup")
def load_model():
    global ocr
    ocr = PaddleOCR(
        use_angle_cls=False,   # IMPORTANT: cls removed in v3
        lang="en"              # change if needed
    )


@app.get("/")
def home():
    return {"status": "OCR API is running 🚀"}


@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    global ocr

    # Read image
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run OCR (NO cls in v3)
    result = ocr.ocr(img)

    # Extract text cleanly
    texts = []
    for line in result[0]:
        texts.append(line[1][0])

    return {
        "text": texts,
        "raw": result
    }
