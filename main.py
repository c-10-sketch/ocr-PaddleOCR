from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
import numpy as np
import cv2

app = FastAPI()

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    show_log=False
)

@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    image_bytes = await file.read()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    result = ocr.ocr(img, cls=True)

    text = []
    if result:
        for line in result:
            for word in line:
                text.append(word[1][0])

    return {"text": " ".join(text)}
