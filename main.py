from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
# Load the SavedModel
model = tf.saved_model.load('../model/1')

# Define the serving signature (adjust as necessary)
infer = model.signatures['serving_default']

class_names = ["Early blight", "Late blight", "Healthy"]

def file_reader(file_content: bytes) -> np.ndarray:
    # Convert bytes to a PIL Image
    image = Image.open(BytesIO(file_content))
    # Convert PIL Image to a numpy array
    image_np = np.array(image)
    return image_np

@app.post("/prediction")
async def prediction(file: UploadFile = File(...)):

    # Read the file and convert it to numpy array
    image_np = file_reader(await file.read())
    image_batch = np.expand_dims(image_np, 0)

    # Perform inference
    input_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    predictions = infer(input_tensor)

    # Get the prediction from the output tensor
    output_tensor = predictions['output_0']  # Use the appropriate key
    predicted_class_index = np.argmax(output_tensor.numpy()[0])
    confidence = np.max(output_tensor.numpy()[0])

    return {
        "predicted_class": class_names[predicted_class_index],
        "confidence": float(confidence)
    }



@app.get("/")
async def get_html():
    with open("model.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
