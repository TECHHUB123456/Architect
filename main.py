from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import replicate
import io


app = FastAPI()

# Enable frontend access (for Android, Web, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

@app.post("/transform")
async def transform_image(
    file: UploadFile = File(...),
    prompt: str = Form("make it modern and furnished")
):
    try:
        image_bytes = await file.read()
        image_io = io.BytesIO(image_bytes)

        output = replicate_client.run(
            "timothybrooks/instruct-pix2pix:6e00c697c5c011ed8f3c1edb3b78b1d2",
            input={"image": image_io, "prompt": prompt}
        )

        return {"output_url": output}

    except Exception as e:
        return {"error": str(e)}
