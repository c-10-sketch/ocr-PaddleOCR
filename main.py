from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

app = FastAPI()

ocr = None

def get_ocr():
    global ocr
    if ocr is None:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en'
        )
    return ocr

@app.get("/")
def home():
    return {"status": "OCR API running"}

@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    ocr = get_ocr()

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
