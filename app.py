"""
Flask API for Footfall Counter
===============================
Upload a video and get footfall counts via REST API.
"""

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import time
import uuid

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model once at startup
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')
print("Model loaded!")


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video_file(video_path, line_position=0.5):
    """Process video and return counts."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    line_y = int(frame_height * line_position)
    
    if fps == 0:
        fps = 30
    
    # Tracking variables
    track_history = defaultdict(lambda: [])
    counted_ids = set()
    entry_count = 0
    exit_count = 0
    
    # Generate output path
    output_id = str(uuid.uuid4())[:8]
    output_path = os.path.join(OUTPUT_FOLDER, f'output_{output_id}.mp4')
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = model.track(frame, persist=True, classes=[0], 
                                verbose=False, conf=0.3, iou=0.5)
            
            # Process detections
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    # Update history
                    track_history[track_id].append((cx, cy))
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)
                    
                    # Check crossing
                    if len(track_history[track_id]) >= 2:
                        prev_y = track_history[track_id][-2][1]
                        curr_y = cy
                        
                        if prev_y < line_y <= curr_y and track_id not in counted_ids:
                            entry_count += 1
                            counted_ids.add(track_id)
                        elif prev_y > line_y >= curr_y and track_id not in counted_ids:
                            exit_count += 1
                            counted_ids.add(track_id)
                    
                    # Draw box
                    color = (0, 255, 0) if cy < line_y else (0, 165, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw counting line
            cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 255), 3)
            
            # Draw counts
            cv2.putText(frame, f"IN: {entry_count}", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT: {exit_count}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            out.write(frame)
    
    finally:
        cap.release()
        out.release()
    
    processing_time = time.time() - start_time
    
    return {
        'entry_count': entry_count,
        'exit_count': exit_count,
        'net_flow': entry_count - exit_count,
        'total_frames': frame_count,
        'video_duration': frame_count / fps,
        'processing_time': processing_time,
        'output_video': output_id,
        'video_properties': {
            'width': frame_width,
            'height': frame_height,
            'fps': fps
        }
    }


@app.route('/')
def index():
    """API documentation."""
    return jsonify({
        'name': 'Footfall Counter API',
        'version': '1.0',
        'endpoints': {
            'POST /count': 'Upload video and get footfall count',
            'GET /download/<output_id>': 'Download processed video',
            'GET /health': 'Check API health'
        },
        'usage': {
            'upload': 'POST /count with form-data: video file and optional line_position (0.0-1.0)',
            'response': 'JSON with entry_count, exit_count, net_flow, and output_video ID'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model': 'loaded'})


@app.route('/count', methods=['POST'])
def count_footfall():
    """Process uploaded video and return counts."""
    
    # Check if file is in request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400
    
    # Get line position (optional)
    line_position = float(request.form.get('line_position', 0.5))
    if not 0 <= line_position <= 1:
        return jsonify({'error': 'line_position must be between 0 and 1'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
        file.save(upload_path)
        
        # Process video
        result = process_video_file(upload_path, line_position)
        
        # Clean up input file
        os.remove(upload_path)
        
        return jsonify({
            'success': True,
            'data': result,
            'download_url': f'/download/{result["output_video"]}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<output_id>')
def download_video(output_id):
    """Download processed video."""
    video_path = os.path.join(OUTPUT_FOLDER, f'output_{output_id}.mp4')
    
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404
    
    return send_file(video_path, as_attachment=True, 
                    download_name=f'footfall_output_{output_id}.mp4')


if __name__ == '__main__':
    print("\n" + "="*60)
    print("FOOTFALL COUNTER API")
    print("="*60)
    print("Starting Flask server on http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /           - API documentation")
    print("  POST /count      - Upload video for counting")
    print("  GET  /download/<id> - Download processed video")
    print("  GET  /health     - Health check")
    print("\nExample usage with curl:")
    print('  curl -X POST -F "video=@input.mp4" -F "line_position=0.5" http://localhost:5000/count')
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)