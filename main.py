from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# Cargar tu modelo YOLOv8 (reemplaza por la ruta si lo nombras diferente)
model = YOLO("best.pt")

class ImageData(BaseModel):
    image: str  # Imagen en base64

@app.post("/detect")
async def detect_image(data: ImageData):
    try:
        # Decodificar imagen base64
        image_bytes = base64.b64decode(data.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Detección con YOLOv8
        results = model(image_np)[0]

        detections = []
        for box in results.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            bbox = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": bbox
            })

        return {"detections": detections}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
