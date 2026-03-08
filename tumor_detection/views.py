from django.shortcuts import render
from django.http import JsonResponse
import os
os.environ['KERAS_BACKEND'] = 'torch'
import keras
import numpy as np
import io
import logging

# Load the model
model = keras.saving.load_model('brain_tumor_model.keras')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Index view for rendering the main HTML page
def index(request):
    return render(request, 'tumor_detection/brain_tumor_index.html')

# Prediction view for processing image and returning prediction
def predict(request):
    if request.method == 'POST' and 'image' in request.FILES:
        try:
            # Get the uploaded image
            img_file = request.FILES['image']
            img_bytes = io.BytesIO(img_file.read())

            # Load the image and preprocess it
            logger.info("Loading and processing the image.")
            img = keras.utils.load_img(img_bytes, target_size=(128, 128))
            img_array = keras.utils.img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]

            # Map the prediction to class names, confidence score, and treatments
            class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
            predicted_label = class_labels[predicted_class]
            confidence_score = float(np.max(predictions) * 100)

            # Treatment recommendations based on the predicted tumor type
            treatment_recommendations = {
                
                'Glioma Tumor': (
                    "STEP1:Surgery: Often the first step to remove as much of the tumor as possible.\n"
                    "STEP2:Radiation Therapy: Commonly used post-surgery to target remaining cells.\n"
                    "STEP3:Chemotherapy: Drugs like temozolomide are sometimes used for aggressive gliomas.\n"
                    "STEP4:Targeted Therapy: Drugs that specifically target genetic mutations or specific pathways in gliomas."
                ),
                'Meningioma Tumor': (
                    "STEP1:Surgery: Primary treatment for accessible meningiomas.\n"
                    "STEP2:Radiation Therapy: Used for meningiomas that cannot be fully removed or are located in sensitive brain areas.\n"
                    "STEP3:Observation: If a meningioma is slow-growing and asymptomatic, doctors may recommend regular monitoring without immediate treatment."
                ),
                'Pituitary Tumor': (
                    "STEP1:Medications: Some pituitary tumors can be managed with drugs that control hormone production.\n"
                    "STEP2:Surgery: Often performed if the tumor is pressing on surrounding tissues.\n"
                    "STEP3:Radiation Therapy: Used for recurrent or residual tumors after surgery."
                ),
                'No Tumor': (
                    "Preventative Measures: Regular health checkups, maintaining a healthy diet, and avoiding exposure to potential carcinogens can contribute to general brain health."
                )
            }
            
            # Generate the diagnosis message with line breaks for treatments
            diagnosis = (
                f"Based on the image, the model suggests a {predicted_label} with a confidence of {confidence_score:.2f}%.\n"
                f"\nRecommended treatments:{treatment_recommendations[predicted_label]}"
            )

            # Return the result as JSON
            return JsonResponse({
                'predicted_class': predicted_label,
                'confidence_score': confidence_score,
                'diagnosis': diagnosis,
            })
        except Exception as e:
            logger.error(f"Error processing the image: {e}")
            return JsonResponse({'error': 'An error occurred while processing the image.'}, status=500)

    return JsonResponse({'error': 'Invalid request method or no image provided'}, status=400)
