from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import paddleocr
import uvicorn


app = FastAPI()

# Initialisation de l'OCR avec PaddleOCR
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="fr")

@app.post("/ocr")
async def extract_text(file: UploadFile = File(...)):
    # Lire le fichier image en mémoire
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(content={"msg": "Erreur lors de la lecture de l'image."}, status_code=400)

    # Exécuter OCR sur l'image
    result = ocr.ocr(img, cls=True)

    if not result or not result[0]:
        return JSONResponse(content={"msg": "Aucun texte détecté."}, status_code=400)
    


    # Extraire le texte détecté
    text = "\n".join([line[1][0] for line in result[0]])

    return {"msg": "Texte extrait avec succès","text": text}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)              