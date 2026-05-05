from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from paddleocr import PaddleOCR

app = FastAPI()

ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    show_log=False
)

@app.get("/")
def home():
    return {"status": "OCR API is running"}

@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    image_bytes = await file.read()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    # 🔥 speed optimization
    img = cv2.resize(img, None, fx=0.6, fy=0.6)

    result = ocr.ocr(img)

    output = []
    for line in result:
        for word_info in line:
            output.append({
                "text": word_info[1][0],
                "confidence": float(word_info[1][1])
            })

    return {"results": output}
