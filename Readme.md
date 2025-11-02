# Footfall Counter using Computer Vision

## Overview
This project implements an AI-powered footfall counter that detects and tracks people in video footage, counting entries and exits across a virtual counting line. The system uses YOLOv8 for person detection and integrated tracking algorithms to maintain consistent identities across frames.

## Approach

### 1. Person Detection
- **Model**: YOLOv8n (nano version for efficient processing)
- **Framework**: Ultralytics YOLO
- **Detection**: Filters for class 0 (person) only
- **Confidence Threshold**: 0.3 to balance accuracy and detection rate
- **IoU Threshold**: 0.5 for non-maximum suppression

### 2. Object Tracking
- **Method**: Built-in YOLOv8 tracking with BoT-SORT algorithm
- **Tracking History**: Maintains last 30 positions per person
- **Unique ID Assignment**: Each person gets a persistent ID throughout the video
- **Trajectory Visualization**: Draws movement paths for better understanding

### 3. Counting Logic
The system implements a virtual horizontal line at a configurable position (default: middle of frame).

**Entry Detection** (Downward crossing):
- Triggered when: `previous_y < line_y ≤ current_y`
- Person moves from above the line to below it

**Exit Detection** (Upward crossing):
- Triggered when: `previous_y > line_y ≥ current_y`
- Person moves from below the line to above it

**Double-counting Prevention**:
- Each tracked ID can only be counted once
- Uses a `counted_ids` set to track already-counted crossings

### 4. Visualization
- **Bounding Boxes**: Green (above line) / Orange (below line)
- **Counting Line**: Yellow horizontal line with label
- **Info Panel**: Real-time display of entries, exits, and active tracks
- **Trajectory Trails**: White lines showing movement history
- **Centroids**: Red dots marking person centers

## Video Source
**Test Video**: `trial_vedio.mp4`
- Resolution: 1920x1080
- Duration: 7.7 seconds
- FPS: 29
- Content: Multiple people moving through a monitored area

The video can be replaced with any video file by updating the `VIDEO_PATH` variable in the configuration section.

## Results

### Final Count
- **Total Entries**: 5
- **Total Exits**: 4
- **Net Flow**: +1
- **Total Line Crossings**: 9

### Performance Metrics
- **Processing Speed**: 3.5 FPS
- **Detection Rate**: 100% of frames had people detected
- **Total Detections**: 2,920
- **Unique Tracks**: 69
- **People Counted**: 9

### Output
- Annotated video saved as: `output_vedio.mp4`
- Video includes bounding boxes, tracking IDs, trajectories, and count display

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/footfall-counter.git
cd footfall-counter
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install opencv-python
pip install ultralytics
pip install numpy
```

The YOLOv8n model will be automatically downloaded on first run (approximately 6MB).

### Step 4: Prepare Your Video
Place your video file in the project directory or update the path in the code:
```python
VIDEO_PATH = r"path/to/your/video.mp4"
```

### Step 5: Run the System
```bash
python main.py
```

## Configuration

Edit these parameters in the `main()` function of `main.py`:

```python
VIDEO_PATH = r"your_video.mp4"      # Input video path
OUTPUT_PATH = 'output_vedio.mp4'    # Output video path
LINE_POSITION = 0.5                 # Line position (0.0=top, 1.0=bottom)
MAX_FRAMES = None                   # Limit frames for testing (None=all)
DISPLAY = False                     # Show video during processing
```

### Advanced Configuration
Modify detection parameters in the `process_video()` method:
```python
results = self.model.track(
    frame, 
    persist=True,
    classes=[0],      # Person class only
    conf=0.3,         # Confidence threshold
    iou=0.5           # IoU threshold
)
```

## Project Structure
```
footfall-counter/
│
├── main.py                 # Main application code
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── trial_vedio.mp4        # Input video (your test video)
├── output_vedio.mp4       # Output video (generated)
└── yolov8n.pt            # YOLO model (auto-downloaded)
```

## Dependencies
```
opencv-python>=4.8.0
ultralytics>=8.0.0
numpy>=1.21.0
```

Create `requirements.txt`:
```bash
pip freeze > requirements.txt
```

## How It Works - Step by Step

1. **Video Loading**: Opens video file and validates properties
2. **Model Initialization**: Loads YOLOv8n for person detection
3. **Frame Processing Loop**:
   - Read frame from video
   - Run YOLO detection + tracking
   - Extract bounding boxes and track IDs
   - Calculate centroids of detected persons
   - Update tracking history for each ID
   - Check for line crossings
   - Increment entry/exit counters
   - Draw visualizations
   - Write annotated frame to output video
4. **Output Generation**: Save processed video with annotations
5. **Statistics**: Display final counts and performance metrics

## Handling Edge Cases

### Occlusions
- Tracking algorithm maintains IDs even during brief occlusions
- History buffer helps maintain trajectory continuity

### Multiple People
- System handles multiple simultaneous detections
- Unique ID per person prevents confusion
- Each ID counted only once

### Noisy Detections
- Confidence threshold filters low-quality detections
- IoU threshold prevents duplicate boxes
- Trajectory smoothing reduces jitter

### Bidirectional Movement
- Separate logic for upward and downward crossings
- Clear distinction between entries and exits

## Troubleshooting

### No People Detected
- Lower confidence threshold (try 0.2)
- Check video quality and lighting
- Ensure people are clearly visible

### No Line Crossings
- Adjust `LINE_POSITION` (try 0.3, 0.5, 0.7)
- Ensure people cross perpendicular to the line
- Check output video to verify line placement

### Slow Processing
- Reduce video resolution
- Use YOLOv8n (already the fastest)
- Set `MAX_FRAMES` for testing shorter segments

### Video Won't Open
- Check file path is correct
- Convert to MP4 with H.264 codec if needed:
  ```bash
  ffmpeg -i input.mp4 -vcodec libx264 output.mp4
  ```

## Bonus Features Implemented ⭐

### 1. Real-time Webcam Processing
**File**: `webcam_counter.py`

Process live video feed from webcam or RTSP stream in real-time.

```bash
python webcam_counter.py
```

**Features**:
- Live webcam support (source=0)
- RTSP stream support (provide URL)
- Real-time FPS display
- Press 'R' to reset counters
- Press 'Q' to quit

### 2. Heatmap Visualization
**File**: `heatmap_counter.py`

Generates activity heatmaps showing where people spend most time.

```bash
python heatmap_counter.py
```

**Output**:
- Annotated video with heatmap overlay
- Final heatmap image (`heatmap_final.jpg`)
- Blue = low activity, Red = high activity

### 3. Flask API
**File**: `app.py`

REST API that accepts video uploads and returns footfall counts.

**Start the API**:
```bash
pip install flask
python app.py
```

**Endpoints**:
- `POST /count` - Upload video for processing
- `GET /download/<id>` - Download processed video
- `GET /health` - Health check
- `GET /` - API documentation

**Example Usage**:
```bash
# Using curl
curl -X POST -F "video=@input.mp4" -F "line_position=0.5" http://localhost:5000/count

# Using Python requests
import requests
files = {'video': open('input.mp4', 'rb')}
data = {'line_position': 0.5}
response = requests.post('http://localhost:5000/count', files=files, data=data)
print(response.json())
```

**Web Interface**:
Open `test_api.html` in a browser for a user-friendly upload interface.

### 4. Handling Occlusions
The system robustly handles occlusions through:
- **Persistent Tracking**: BoT-SORT algorithm maintains IDs during brief occlusions
- **Trajectory History**: 30-frame history buffer smooths tracking
- **Re-identification**: Tracks can be recovered after occlusion
- **Confidence Filtering**: Low-confidence detections filtered out

## Future Enhancements

- [ ] Multi-line counting zones
- [ ] Age/gender classification
- [ ] Dwell time analysis
- [ ] Cloud deployment support
- [ ] Mobile app integration
- [ ] Dashboard with analytics
- [ ] Email alerts for threshold violations

## Performance Optimization Tips

1. **Use GPU**: Install CUDA and cuDNN for 10-20x speedup
2. **Batch Processing**: Process multiple frames simultaneously
3. **Frame Skipping**: Process every Nth frame for faster results
4. **Lower Resolution**: Resize input frames if acceptable
5. **Model Selection**: Use YOLOv8n (fastest) vs YOLOv8x (most accurate)

## License
This project is open-source and available under the MIT License.

## Author
[Anuj Mishra]

## Acknowledgments
- Ultralytics for YOLOv8
- OpenCV community
- BoT-SORT tracking algorithm developers

## Contact
For questions or issues, please open an issue on GitHub or contact [anujmishra77386@gmail.com]

---

**Note**: This system is designed for educational and research purposes. For production deployment, consider additional error handling, logging, and security measures.