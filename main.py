from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import replicate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

replicate_client = replicate.Client(api_token="r8_3yW7As146lHaVTgGWNR6IdETxeQZFEA2Cb8Gb")

@app.post("/transform")
async def transform_image(prompt: str = Form(...), file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        output = replicate_client.run(
            "timbrooks/instruct-pix2pix:4e15f8c29d292d49b9bd06c6800c9529706eaf490396efcf6c0d371c33b95e04",
            input={"image": image_bytes, "prompt": prompt}
        )
        return {"output_url": output}
    except replicate.exceptions.ReplicateError as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
