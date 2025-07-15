from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o ["http://localhost:8100"] si lo prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")

class ImageData(BaseModel):
    image: str  # Imagen en base64

@app.post("/detect")
async def detect_image(data: ImageData):
    try:
        # Decodificar imagen base64
        image_bytes = base64.b64decode(data.image.split(",")[-1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Detección
        results = model(image_np)[0]

        detections = []
        annotated_image = results.plot()  # Dibujar cajas y etiquetas
        annotated_pil = Image.fromarray(annotated_image)

        for box in results.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            bbox = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": bbox
            })

        # Codificar la imagen con cajas en base64
        buffered = io.BytesIO()
        annotated_pil.save(buffered, format="JPEG")
        annotated_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "detections": detections,
            "image": f"data:image/jpeg;base64,{annotated_base64}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
