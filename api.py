import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from predict import predict
import numpy as np

app = FastAPI(
    title="Heart Sound Classifier API",
    description="REST API backend for the Heart Sound GAN-Classifier. Returns classification probabilities and segment voting."
)

# Enable CORS (Cross-Origin Resource Sharing) so Lovable frontends can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_heart_sound(file: UploadFile = File(...)):
    """
    Receives an uploaded WAV file, runs the signal classification pipeline, 
    and returns a structured JSON payload compatible with frontend dashboards.
    """
    temp_path = f"temp_{file.filename}"
    # Save the uploaded file temporarily
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        # Run prediction pipeline using predict.py functions
        result = predict(temp_path)
        
        # Calculate granular segment votes and probabilities
        segment_probs = np.array(result['segment_probs'])
        normal_segments = int(np.sum(segment_probs < 0.5))
        abnormal_segments = int(np.sum(segment_probs >= 0.5))
        avg_prob = float(np.mean(segment_probs))
        
        abnormal_prob = avg_prob * 100
        normal_prob = (1.0 - avg_prob) * 100
        confidence = max(normal_prob, abnormal_prob)
        
        # Determine status
        if confidence < 60.0:
            pred_status = "uncertain"
        elif abnormal_prob >= 50.0:
            pred_status = "abnormal"
        else:
            pred_status = "normal"
            
        # Return response matching the Lovable schema exactly
        response = {
            "file_name": file.filename,
            "global_prediction": result['class'],
            "global_confidence_percent": round(confidence, 2),
            "normal_probability_percent": round(normal_prob, 2),
            "abnormal_probability_percent": round(abnormal_prob, 2),
            "prediction_status": pred_status,
            "feature_dimensions": {
                "time_frames": 64,
                "features": 39,
                "total_segments": result['num_segments']
            },
            "segment_voting": {
                "total_segments": result['num_segments'],
                "normal_segments": normal_segments,
                "abnormal_segments": abnormal_segments
            }
        }
        return response
        
    except Exception as e:
        return {"error": str(e)}
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

if __name__ == '__main__':
    # Start the API server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
