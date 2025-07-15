from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Para mayor seguridad, puedes reemplazar por ["http://localhost:8100"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint raíz para verificar funcionamiento
@app.get("/")
async def root():
    return {"message": "API de detección de residuos funcionando correctamente"}

# Cargar el modelo YOLOv8
model = YOLO("best.pt")

# Modelo de entrada
class ImageData(BaseModel):
    image: str  # Imagen en base64

@app.post("/detect")
async def detect_image(data: ImageData):
    try:
        # Eliminar encabezado base64 si existe
        if "," in data.image:
            base64_data = data.image.split(",")[1]
        else:
            base64_data = data.image

        # Decodificar y convertir la imagen
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Ejecutar el modelo YOLO
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
        raise HTTPException(status_code=500, detail=f"Error en detección: {str(e)}")
