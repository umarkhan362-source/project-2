from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import io
app = FastAPI(title="Document Classification API", description='OCR, Document Classification, and Information Extraction API', version="1.0.0")

# Load the pre-trained model
vectorizer = joblib.load('vectorizer.pkl')
classifier = joblib.load('classifier.pkl')

@app.get("/")
def root():
    return {'message': 'Welcome to the Document Classification API. Use the /classify endpoint to classify your documents.', 'version': '1.0.0',
            'endpoints': ['/classify', '/extract', '/process']}

@app.post("/classify")
async def classify_document(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        # Perform OCR to extract text from the image
        text = pytesseract.image_to_string(image)
        # Preprocess the extracted text and convert it to a feature vector
        # vectorize and classify
        text_vec = vectorizer.transform([text])
        prediction = classifier.predict(text_vec)[0]
        probabilities = classifier.predict_proba(text_vec)[0]
        confidence = max(probabilities)
        return {'document_types': prediction, 'confidence': float(confidence), 'all_probabilities': {cls: float(prob) for cls, prob in zip(classifier.classes_, probabilities)}}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

from extractors import extract_dates, extract_amounts, extract_entities

@app.post("/extract")
async def extract_information(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        text = pytesseract.image_to_string(image)
        dates = extract_dates(text)
        amounts = extract_amounts(text)
        entities = extract_entities(text)
        return {'dates': dates, 'amounts': amounts, 'entities': entities, 'raw_text': text[:500]}  # Return a snippet of the raw text for reference
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        text = pytesseract.image_to_string(image)
        text_vec = vectorizer.transform([text])
        doc_type = classifier.predict(text_vec)[0]
        confidence = max(classifier.predict_proba(text_vec)[0])
        extracted_data = {'dates': extract_dates(text), 'amounts': extract_amounts(text), 'entities': extract_entities(text)}
        return {'document_type': doc_type, 'confidence': float(confidence), 'extracted_data': extracted_data, 'status': 'success'}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e), 'status': 'failed'})

