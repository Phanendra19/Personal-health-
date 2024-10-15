from flask import Flask, render_template, request, jsonify
import openai
import requests  # For calling external APIs
import torch
from transformers import VisionEncoderDecoderModel, AutoProcessor, pipeline
from PIL import Image
import os
import subprocess
import boto3  # AWS HealthLake Integration
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import torchvision.transforms as transforms
from torchvision import models

app = Flask(__name__)

# Set OpenAI API Key for GPT-4
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load a Pretrained VisionEncoderDecoder Model for Multimodal Tasks (e.g., Image + Text)
# multimodal_model = VisionEncoderDecoderModel.from_pretrained("google/vit-gpt2-coco-en")
# processor = AutoProcessor.from_pretrained("google/vit-gpt2-coco-en")
try:
    multimodal_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = AutoProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # Initialize Stable Diffusion model for image generation
    image_gen_pipeline = pipeline("text-to-image", model="stabilityai/stable-diffusion-2-1-base")


    # Pre-trained ResNet Model for Medical Image Classification
    resnet_model = models.resnet50(pretrained=True)

    # Image Preprocessing for ResNet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # AWS HealthLake Configuration
    client = boto3.client('healthlake', region_name='us-west-2')  # Change to your AWS region

    # Google Places API Key for hospital search
    GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
except Exception as e:
    print(f"Error loading model: {e}")
# Serve the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Serve the disease diagnosis page
@app.route('/disease.html')
def disease_page():
    return render_template('disease.html')

# Serve the first aid page
@app.route('/firstaid.html')
def firstaid_page():
    return render_template('firstaid.html')

# Serve the motivation page
@app.route('/motivation.html')
def motivation_page():
    return render_template('motivation.html')

# Serve the medicine analysis page
@app.route('/imageanalysis.html')
def medicine_page():
    return render_template('imageanaysis.html')

# Function to suggest hospitals based on location and condition
def suggest_hospitals(location, condition):
    search_query = f"{condition} hospital"
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&radius=5000&type=hospital&keyword={search_query}&key={GOOGLE_PLACES_API_KEY}"
    
    response = requests.get(url)
    hospitals = response.json()
    
    if 'results' in hospitals:
        return [{'name': hospital['name'], 'address': hospital.get('vicinity', 'N/A')} for hospital in hospitals['results']]
    else:
        return None

# Function to store FHIR data in AWS HealthLake
def store_fhir_data(fhir_data):
    try:
        response = client.start_fhir_import_job(
            DataAccessRoleArn='arn:aws:iam::575108948712:role/HealthLakeAccessRole',
            InputDataConfig={'S3Uri': 's3://your-healthlake-bucket/fhir-data/'},
            JobName='ImportFHIRDataJob',
            OutputDataConfig={'S3Uri': 's3://your-healthlake-bucket/output/'}
        )
        return response
    except (NoCredentialsError, PartialCredentialsError) as e:
        return {'error': str(e)}

# Function to retrieve patient data from AWS HealthLake
def get_patient_data(patient_id):
    try:
        response = client.search(
            resourceType='Patient',
            searchParameters={'identifier': patient_id}
        )
        return response
    except Exception as e:
        return {'error': str(e)}

# Directory to save generated images for video
IMAGE_DIR = "first_aid_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

### Disease Diagnosis and Hospital Suggestion ###
@app.route('/diagnose_disease', methods=['POST'])
def diagnose_disease():
    data = request.get_json()
    symptoms_text = data['symptoms']
    user_location = data.get('location', None)  # User's location (latitude, longitude or city name)

    # Use GPT-4 to diagnose the disease
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=f"Based on the symptoms: {symptoms_text}, provide a diagnosis and suggest treatment options.",
        max_tokens=100
    )
    diagnosis = response.choices[0].text.strip()

    # Suggest nearby hospitals if location is provided
    hospitals = None
    if user_location:
        hospitals = suggest_hospitals(user_location, diagnosis)

    # Create FHIR-compatible data and store in AWS HealthLake
    fhir_data = {
        "resourceType": "Condition",
        "clinicalStatus": "active",
        "code": {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "disease_code_here",
                    "display": diagnosis
                }
            ]
        },
        "subject": {
            "reference": f"Patient/{data['patient_id']}"
        }
    }

    store_fhir_data(fhir_data)

    response_data = {'diagnosis': diagnosis}
    if hospitals:
        response_data['hospitals'] = hospitals

    return jsonify(response_data)

### Retrieve Patient Data from AWS HealthLake ###
@app.route('/get_patient_data/<patient_id>', methods=['GET'])
def fetch_patient_data(patient_id):
    patient_data = get_patient_data(patient_id)
    if 'error' in patient_data:
        return jsonify({'status': 'error', 'message': patient_data['error']}), 500
    return jsonify({'status': 'success', 'patient_data': patient_data}), 200

### Generate First Aid Images and Video ###
@app.route('/generate_first_aid', methods=['POST'])
def generate_first_aid_images_and_video():
    data = request.get_json()
    condition = data['condition']

    # Ask GPT-4 for the first aid steps
    gpt4_response = openai.Completion.create(
        engine="gpt-4",
        prompt=f"Provide first aid steps for a person experiencing {condition}.",
        max_tokens=200
    )
    first_aid_steps = gpt4_response.choices[0].text.strip().split("\n")

    # Generate images for each step
    generated_images = []
    for i, step in enumerate(first_aid_steps):
        image = image_gen_pipeline(step)[0]
        image_path = os.path.join(IMAGE_DIR, f"first_aid_step_{i+1}.png")
        image.save(image_path)
        generated_images.append(image_path)

    # Create video from the generated images
    video_path = create_video_from_images(generated_images, "first_aid_video.mp4")

    # Return the steps and the video path
    return jsonify({
        'steps': first_aid_steps,
        'video_path': video_path
    })

### Motivational Message for Depressed Users ###
@app.route('/motivate_user', methods=['POST'])
def motivate_depressed_user():
    data = request.get_json()
    depression_message = data['message']

    response = openai.Completion.create(
        engine="gpt-4",
        prompt=f"Respond empathetically to the following statement: {depression_message}.",
        max_tokens=100
    )
    motivational_message = response.choices[0].text.strip()
    return jsonify({'motivation': motivational_message})

### Medicine Image Analysis ###
@app.route('/analyze_medicine', methods=['POST'])
def analyze_medicine_image():
    image_file = request.files['image']
    image_file.save('uploaded_image.jpg')


    image = Image.open('uploaded_image.jpg')
    inputs = processor(images=image, return_tensors="pt")
    outputs = multimodal_model.generate(**inputs)
    medication_description = processor.decode(outputs[0], skip_special_tokens=True)

    # Further analysis with GPT-4
    response_med = openai.Completion.create(
        engine="gpt-4",
        prompt=f"Describe the uses, dosage, and side effects of the medication based on the description: {medication_description}.",
        max_tokens=150
    )
    medication_info = response_med.choices[0].text.strip()
    return jsonify({'medication_info': medication_info})

# Start Flask application
if __name__ == '__main__':
    app.run(debug=True)