from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates for HTML
templates = Jinja2Templates(directory="templates")

# Load the trained model
model = None
try:
    model = tf.keras.models.load_model("bestModel.keras")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the model.")

# Define class names
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Disease treatment knowledge base
disease_treatment = {
    "Tomato___Bacterial_spot": {
        "description": "A bacterial disease causing dark, wet spots on leaves and fruits.",
        "treatment": [
            "Apply copper-based bactericides.",
            "Remove and destroy infected plants.",
            "Avoid overhead watering."
        ]
    },
    "Tomato___Early_blight": {
        "description": "A fungal disease causing dark spots with concentric rings on leaves.",
        "treatment": [
            "Apply fungicides containing chlorothalonil or mancozeb.",
            "Remove infected leaves and plants.",
            "Rotate crops to prevent soil-borne fungi."
        ]
    },
    "Tomato___Late_blight": {
        "description": "A fungal disease causing dark, water-soaked lesions on leaves and stems.",
        "treatment": [
            "Apply fungicides containing chlorothalonil or mancozeb.",
            "Remove and destroy infected plants.",
            "Avoid overhead watering and ensure proper spacing for air circulation."
        ]
    },
    "Tomato___Leaf_Mold": {
        "description": "A fungal disease causing yellow spots on the upper leaf surface and mold on the underside.",
        "treatment": [
            "Apply fungicides containing chlorothalonil or mancozeb.",
            "Improve air circulation by spacing plants properly.",
            "Avoid overhead watering."
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "A fungal disease causing small, circular spots with gray centers and dark edges on leaves.",
        "treatment": [
            "Apply fungicides containing chlorothalonil or mancozeb.",
            "Remove infected leaves and destroy them.",
            "Avoid overhead watering."
        ]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "A pest infestation causing yellow stippling on leaves and fine webbing on the plant.",
        "treatment": [
            "Apply insecticidal soap or neem oil.",
            "Increase humidity to deter mites.",
            "Remove heavily infested leaves."
        ]
    },
    "Tomato___Target_Spot": {
        "description": "A fungal disease causing dark, target-like spots on leaves.",
        "treatment": [
            "Apply fungicides containing chlorothalonil or mancozeb.",
            "Remove infected leaves and destroy them.",
            "Avoid overhead watering."
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "A viral disease causing yellowing and curling of leaves, stunted growth, and reduced fruit production.",
        "treatment": [
            "Remove and destroy infected plants.",
            "Control whitefly populations (the virus vector) using insecticides.",
            "Plant resistant varieties if available."
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "A viral disease causing mottled leaves, stunted growth, and reduced fruit yield.",
        "treatment": [
            "Remove and destroy infected plants.",
            "Disinfect tools and hands to prevent spread.",
            "Plant resistant varieties if available."
        ]
    },
    "Tomato___healthy": {
        "description": "The plant is healthy and shows no signs of disease.",
        "treatment": [
            "Continue good cultural practices.",
            "Monitor for early signs of disease.",
            "Maintain proper watering and fertilization."
        ]
    }
}

# Function to preprocess the image
def preprocess_image(image: Image.Image) -> np.ndarray:
    try:
        image = image.convert("RGB")  # Ensure RGB mode
        image = image.resize((256, 256))  # Resize to match model input size
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Image preprocessing failed.")

# Function to check if the image is a tomato leaf
def is_tomato_leaf(image: Image.Image) -> bool:
    """
    Improved heuristic to check if the image is a tomato leaf.
    """
    # Convert image to numpy array
    image_array = np.array(image)

    # Check if the dominant color is green
    green_pixels = np.sum((image_array[:, :, 1] > image_array[:, :, 0]) & (image_array[:, :, 1] > image_array[:, :, 2]))
    total_pixels = image_array.shape[0] * image_array.shape[1]
    green_ratio = green_pixels / total_pixels

    # Log green ratio for debugging
    logger.info(f"Green pixel ratio: {green_ratio}")

    # Check if the image contains enough green pixels
    if green_ratio < 0.4:  # Lowered threshold to 40%
        logger.info("Image rejected: Not enough green pixels.")
        return False

    # Additional check: Ensure the image is not too dark or too bright
    brightness = np.mean(image_array) / 255.0
    logger.info(f"Brightness: {brightness}")

    if brightness < 0.2 or brightness > 0.8:
        logger.info("Image rejected: Too dark or too bright.")
        return False

    return True

# Function to suggest treatment
def suggest_treatment(predicted_class: str) -> dict:
    return disease_treatment.get(predicted_class, {
        "description": "No detailed information available.",
        "treatment": ["No treatment information available."]
    })

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Validate file size (e.g., 5MB limit)
    max_file_size = 5 * 1024 * 1024  # 5MB
    file_contents = await file.read()
    if len(file_contents) > max_file_size:
        raise HTTPException(status_code=400, detail="File size exceeds the 5MB limit.")

    if not file_contents:
        raise HTTPException(status_code=400, detail="File is empty.")

    # Log file type before validation
    logger.info(f"Received file: {file.filename} ({file.content_type})")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPG or PNG image.")

    try:
        # Read the image
        image = Image.open(io.BytesIO(file_contents))

        # Check if the image is a tomato leaf using shape-based detection
        if not is_tomato_leaf(image):
            raise HTTPException(status_code=400, detail="The uploaded image does not appear to be a tomato leaf. Please upload a valid tomato leaf image.")

        # Preprocess the image and make a prediction
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))

        # Log raw prediction probabilities
        logger.info(f"Prediction probabilities: {prediction}")
        logger.info(f"Predicted class: {predicted_class}")
        logger.info(f"Confidence: {confidence}")

        # Get treatment suggestion
        treatment_info = suggest_treatment(predicted_class)

        # Return prediction and treatment
        return JSONResponse({
            "predicted_disease": predicted_class,
            "confidence": confidence,
            "description": treatment_info["description"],
            "treatment": treatment_info["treatment"]
        })
    except UnidentifiedImageError:
        logger.error("Unable to process the image. The file might be corrupt.")
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid JPG or PNG image.")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="We encountered an issue while processing your request. Please ensure the image is clear and try again.")
   

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)