from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from paddleocr import PaddleOCR

app = FastAPI()

# Load model ONCE at startup (very important for Render stability)
ocr = PaddleOCR(use_angle_cls=False, lang='en')

@app.get("/")
def home():
    return {"status": "OCR API is running"}

@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()

    # Convert to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run OCR (NO cls, NO show_log)
    result = ocr.ocr(img)

    # Format response cleanly
    output = []
    for line in result:
        for word_info in line:
            text = word_info[1][0]
            confidence = float(word_info[1][1])
            output.append({
                "text": text,
                "confidence": confidence
            })

    return {
        "results": output
    }
