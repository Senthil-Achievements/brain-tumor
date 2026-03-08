import os
os.environ['KERAS_BACKEND'] = 'torch'
import keras
import numpy as np
import io
import gradio as gr

# Load the model directly
model = keras.saving.load_model('brain_tumor_model.keras')

class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

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

def predict_tumor(image_path):
    if image_path is None:
        return "Please upload an MRI image.", 0.0, ""

    # Load the image and preprocess it
    img = keras.utils.load_img(image_path, target_size=(128, 128))
    img_array = keras.utils.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    predicted_label = class_labels[predicted_class]
    confidence_score = float(np.max(predictions) * 100)
    
    treatment = treatment_recommendations[predicted_label]
    
    diagnosis = f"Based on the image, the model diagnosis is a {predicted_label}."
    
    return diagnosis, confidence_score, treatment

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# Brain Tumor Prediction")
    gr.Markdown("Upload an MRI scan to predict the type of tumor and receive a diagnosis.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload MRI Image")
            submit_btn = gr.Button("Predict", variant="primary")
        
        with gr.Column():
            output_diagnosis = gr.Textbox(label="Diagnosis")
            output_confidence = gr.Number(label="Confidence Score (%)")
            output_treatment = gr.Textbox(label="Recommended Treatments", lines=5)
            
    submit_btn.click(
        fn=predict_tumor,
        inputs=[image_input],
        outputs=[output_diagnosis, output_confidence, output_treatment]
    )

if __name__ == "__main__":
    interface.launch()
