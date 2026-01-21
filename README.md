# Face Recognition System

A real-time face recognition system using DeepFace library with ArcFace model and RetinaFace detector. This project provides webcam-based face detection, recognition, and database management.

## Features

- 🎥 **Real-time Face Recognition**: Detect and recognize faces from webcam feed
- 🔍 **High Accuracy**: Uses ArcFace model with 99.41% accuracy on LFW dataset
- 📸 **Easy Database Management**: Add new faces to the database using your camera
- ⚡ **Optimized Performance**: Recognition runs every 0.5 seconds for smooth video
- 🎯 **RetinaFace Detector**: State-of-the-art face detection backend

## Technology Stack

- **Model**: ArcFace (512D embeddings)
- **Detector**: RetinaFace
- **Distance Metric**: Cosine Similarity
- **Threshold**: 0.6
- **Framework**: DeepFace, OpenCV, TensorFlow

## Project Structure

```
face_recognition/
├── face_recognition.py    # Main face recognition script
├── add_to_db.py          # Add faces to database
├── faces_db/             # Face database directory
├── requirements.txt      # Project dependencies
├── .gitignore           # Git ignore file
└── README.md            # Project documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows/Linux/macOS

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/face-recognition.git
   cd face-recognition
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Create face database directory**
   ```bash
   mkdir faces_db
   ```

## Usage

### 1. Add Faces to Database

Run the database management script to add new faces:

```bash
python add_to_db.py
```

**Controls:**
- Press `a` to capture and add a face
- Enter the person's name when prompted
- Press `q` to quit

**Note**: Face images are saved as `faces_db/{name}.jpg`

### 2. Run Face Recognition

Start the face recognition system:

```bash
python face_recognition.py
```

**Controls:**
- The system will automatically detect and recognize faces
- Press `q` to quit

**Features:**
- Green text shows recognized person's name
- Blue rectangle highlights the detected face
- "Unknown" label for unrecognized faces

## Configuration

You can modify these parameters in `face_recognition.py`:

```python
MODEL = "ArcFace"           # Face recognition model
DETECTOR = "retinaface"     # Face detection backend
Threshold = 0.6             # Recognition threshold (lower = stricter)
run_every_seconds = 0.5     # Recognition frequency
```

### Available Models

| Model | Accuracy | Speed | Embedding Size |
|-------|----------|-------|----------------|
| ArcFace | 99.41% | Medium | 512D |
| Facenet512 | 99.65% | Medium | 512D |
| Facenet | 99.20% | Fast | 128D |
| VGG-Face | 98.78% | Slow | 2622D |
| OpenFace | 93.80% | Very Fast | 128D |

### Available Detectors

| Detector | Accuracy | Speed |
|----------|----------|-------|
| retinaface | Highest | Medium |
| mtcnn | High | Medium |
| mediapipe | High | Very Fast |
| opencv | Medium | Fastest |
| ssd | Medium-High | Fast |

## How It Works

1. **Face Detection**: RetinaFace detector locates faces in the video frame
2. **Feature Extraction**: ArcFace model generates 512D embeddings for each face
3. **Face Matching**: Compares embeddings with database using cosine similarity
4. **Recognition**: Returns the best match if distance < threshold (0.6)

## Performance

- **Recognition Speed**: ~2 FPS (recognition every 0.5s)
- **Model Accuracy**: 99.41% on LFW dataset
- **Distance Metric**: Cosine similarity
- **Threshold**: 0.6 (lower = stricter matching)

## Troubleshooting

### Camera Not Opening
```python
# Check available cameras
cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
```

### Model Download Issues
- Models are downloaded automatically on first run
- Check internet connection
- Models stored in `.deepface/weights/`

### Low Recognition Accuracy
- Adjust threshold: `Threshold = 0.7` (less strict)
- Use better lighting conditions
- Add multiple images per person to database
- Ensure face images are clear and front-facing

### Slow Performance
- Change to faster model: `MODEL = "Facenet"`
- Use faster detector: `DETECTOR = "mediapipe"`
- Increase recognition interval: `run_every_seconds = 1.0`

## Dependencies

- deepface==0.0.92
- opencv-python==4.10.0.84
- tensorflow==2.15.0
- numpy==1.26.4
- pandas==2.2.2
- retina-face==0.0.17
- mtcnn==0.1.1
- mediapipe==0.10.14

See `requirements.txt` for complete list.

## Resources

- [DeepFace GitHub](https://github.com/serengil/deepface)
- [ArcFace Paper](https://arxiv.org/abs/1801.07698)
- [Model Benchmarks](https://github.com/serengil/deepface#face-recognition-models)
- [DeepFace Documentation](https://github.com/serengil/deepface/tree/master/deepface)




## Acknowledgments

- DeepFace library by Sefik Ilkin Serengil
- ArcFace model by Jiankang Deng et al.
- RetinaFace detector by Jiankang Deng et al.

