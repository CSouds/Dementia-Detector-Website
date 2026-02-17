from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from datetime import datetime
import json
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app)

# Archia Configuration
ARCHIA_TOKEN = "ask_U5slEz8M-QXZQ0Vmn3PKloJ-mTGnOC5G52Eeyjv623E="
BASE_URL = "https://registry.archia.app/v1"

def transcribe_text(file):
    # print(file)

    model = WhisperModel(
    "base",
    device="cpu",
    compute_type="float32"
    )

    segments, info = model.transcribe(
        file,
        word_timestamps=True
    )

    pause_threshold = 0.5
    output = []
    previous_end = None

    for segment in segments:
        for word in segment.words:
            if previous_end is not None:
                gap = word.start - previous_end
                if gap >= pause_threshold:
                    output.append(f"(pause for {gap:.1f}s)")
            output.append(word.word)
            previous_end = word.end

    text = " ".join(output)
    text = text.replace(" ,", ",").replace(" .", ".").replace("  ", " ")
    # print(text)

    return text

def call_archia_agent(prompt):
    """Call the Archia Dementia Detector agent"""
    headers = {
        "Authorization": f"Bearer {ARCHIA_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "agent:Dementia Detector",
        "input": prompt,
        "stream": False
    }
    
    response = requests.post(f"{BASE_URL}/responses", headers=headers, json=payload)
    return response.json()

# Create directories for uploads and analysis results
os.makedirs('../uploads', exist_ok=True)
os.makedirs('../analysis_results', exist_ok=True)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if the backend is running"""
    return jsonify({
        'status': 'healthy',
        'message': 'Backend is running'
    })

@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    """Handle file uploads and extract text"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        # Read file content
        if file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        elif file.filename.endswith('.wav') or file.filename.endswith('.mp3'):
            text = transcribe_text(file)
        elif file.filename.endswith('.doc') or file.filename.endswith('.docx'):
            return jsonify({
                'success': False,
                'message': 'Please convert your document to .txt format'
            }), 400
        else:
            return jsonify({
                'success': False,
                'message': 'Unsupported file format. Please use .txt, .wav, or .mp3 files'
            }), 400
        
        # Save uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"upload_{timestamp}_{file.filename}"
        filepath = os.path.join('../uploads', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return jsonify({
            'success': True,
            'text': text,
            'filename': filename
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing file: {str(e)}'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Analyze text using Archia Dementia Detector agent"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                'success': False,
                'message': 'No text provided for analysis'
            }), 400
        
        # Count words
        word_count = len(text.split())
        if word_count < 150:
            return jsonify({
                'success': False,
                'message': f'Text too short. Please provide at least 150 words (current: {word_count})'
            }), 400
        
        # Create the analysis prompt
        prompt = text

        # Call Archia Agent
        response = call_archia_agent(prompt)
        
        # Extract the analysis text from the response
        # Adjust this based on the actual response structure from Archia
        analysis = response['output'][0]['content'][0]['text']
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'word_count': word_count
        })
    
    except requests.RequestException as e:
        return jsonify({
            'success': False,
            'message': f'Archia API Error: {str(e)}'
        }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during analysis: {str(e)}'
        }), 500

@app.route('/api/save-analysis', methods=['POST'])
def save_analysis():
    """Save analysis results to a file"""
    try:
        data = request.json
        text = data.get('text', '')
        analysis = data.get('analysis', '')
        user_id = data.get('user_id', 'anonymous')
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"analysis_{user_id}_{timestamp}.json"
        filepath = os.path.join('../analysis_results', filename)
        
        # Prepare data to save
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'original_text': text,
            'analysis': analysis,
            'word_count': len(text.split())
        }
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error saving analysis: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SpeechCare AI Backend Server Starting...")
    print("Using Archia Dementia Detector Agent")
    print("Backend will run on: http://localhost:5000")
    print("Make sure your frontend is configured to use this URL")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5001, host='0.0.0.0')