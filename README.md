# susDetect: Market Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue.svg)](https://www.typescriptlang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

An AI-powered system for detecting market manipulation patterns (spoofing and layering) in financial order book data using LSTM neural networks.

## 🎯 Overview

susDetect uses deep learning to analyze order book dynamics and identify anomalous trading patterns that may indicate market manipulation. The system processes sequences of order book snapshots to detect:

- **Spoofing**: Placing large orders with no intention to execute, creating false market signals
- **Layering**: Building multiple orders at different price levels to manipulate market perception
- **Normal Activity**: Legitimate trading patterns

## 🚀 Features

### Core Functionality

- **LSTM-based Detection**: Deep learning model trained on synthetic order book data
- **Real-time Analysis**: Process 100-snapshot windows for immediate anomaly detection
- **Confidence Scoring**: Quantitative confidence levels for each prediction
- **Interactive Visualization**: Web interface for data exploration and testing

### Data Processing

- **Synthetic Data Generation**: Create realistic order book scenarios with planted anomalies
- **Flexible Input**: Support for both generated and user-provided data
- **Order Book Modeling**: 6-level order book representation (3 bid + 3 ask levels)

### User Interface

- **Web Dashboard**: Modern React/Next.js interface
- **Data Browser**: Interactive snapshot navigation and visualization
- **Real-time Feedback**: Live prediction results with detailed explanations

## 🏗️ Architecture

```
susDetect/
├── backend/                 # Python FastAPI server
│   ├── server.py           # Main API server
│   ├── model.py            # LSTM model definition
│   ├── data_gen.py         # Data generation utilities
│   └── main_model.pt       # Trained model weights
├── frontend/               # Next.js web application
│   └── src/app/page.tsx    # Main dashboard component
├── test.ipynb             # Jupyter notebook for testing
└── README.md              # This file
```

### Model Architecture

- **Input**: 100 × 18 tensor (100 time steps × 6 orders × 3 features each)
- **LSTM Layer**: 64 hidden units, 1 layer, batch_first=True
- **Output**: Binary classification (0=normal, 1=spoofing) with confidence score
- **Loss Function**: BCEWithLogitsLoss
- **Optimizer**: Adam (lr=0.001)

## 📋 Prerequisites

- **Python 3.8+**
- **Node.js 18+**
- **PyTorch 2.0+**
- **FastAPI**
- **Uvicorn**

## 🛠️ Installation

### Backend Setup

1. **Create Conda Environment** (recommended):

```bash
conda create -n susdetect python=3.10
conda activate susdetect
```

2. **Install Dependencies**:

```bash
pip install torch torchvision torchaudio
pip install fastapi uvicorn
pip install numpy scikit-learn
```

3. **Verify Installation**:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### Frontend Setup

1. **Navigate to Frontend Directory**:

```bash
cd frontend
```

2. **Install Dependencies**:

```bash
npm install
```

3. **Start Development Server**:

```bash
npm run dev
```

## 🚀 Usage

### Starting the Application

1. **Start Backend Server**:

```bash
# From project root
python server.py
# or with uvicorn
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

2. **Start Frontend** (in another terminal):

```bash
cd frontend
npm run dev
```

3. **Access Application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Using the Web Interface

#### Generate Test Data

1. Click "Generate dummy data" tab
2. Click "Generate window" button
3. Browse through 100 snapshots using the slider
4. Click "Run detection" to analyze

#### Upload Custom Data

1. Click "Paste your own" tab
2. Input JSON array of 100 snapshots
3. Each snapshot: array of 6 orders × [price, quantity, cancel_speed]
4. Click "Load data" then "Run detection"

### API Usage

#### Prediction Endpoint

```bash
POST /predict
Content-Type: application/json

{
  "data": [
    [[price1, qty1, speed1], [price2, qty2, speed2], ...],  // 6 orders
    [...],  // 99 more snapshots
  ]
}
```

**Response**:

```json
{
  "Prediction": 0,
  "Confidence": 0.87
}
```

## 📊 Data Format

### Order Book Snapshot Structure

Each snapshot represents 6 orders (3 bid + 3 ask):

```javascript
[
  [price, quantity, cancel_speed], // Order 1
  [price, quantity, cancel_speed], // Order 2
  // ... 4 more orders
];
```

- **Price**: Positive for bids, negative for asks
- **Quantity**: Order size
- **Cancel Speed**: 0.0-1.0 (higher = faster cancellations)

### Window Format

100 consecutive snapshots forming a time series:

```javascript
[
  [snapshot_1], [snapshot_2], ..., [snapshot_100]
]
```

## 🧠 Model Training

### Training Data Generation

```python
from data_gen import generate_data

# Generate 1000 samples (500 normal + 500 spoofing)
X, y = generate_data(mid_price=1000, n=1000)
```

### Training Process

```python
from model import Anomaly_Detect

model = Anomaly_Detect()
# Training loop with BCEWithLogitsLoss + Adam optimizer
# 100 epochs, batch_size=32
```

### Model Evaluation

- **Accuracy**: ~75% on test set
- **Precision**: 0.62 (spoofing detection)
- **Recall**: 0.36 (spoofing detection)

## 🔧 Configuration

### Model Hyperparameters

```python
INPUT_SIZE = 18      # 6 orders × 3 features
HIDDEN_SIZE = 64     # LSTM hidden units
NUM_LAYERS = 1       # LSTM layers
OUTPUT_SIZE = 1      # Binary classification
```

### Training Parameters

```python
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
WINDOW_SIZE = 100
```

## 🧪 Testing

### Jupyter Notebook

```bash
jupyter notebook test.ipynb
```

The notebook includes:

- Data generation examples
- Model training walkthrough
- Evaluation metrics
- Visualization of results

### API Testing

```bash
# Test with curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d @test_data.json
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 for Python code
- Use TypeScript strict mode
- Add tests for new features
- Update documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch** for the deep learning framework
- **FastAPI** for the web API framework
- **Next.js** for the React framework
- **Tailwind CSS** for styling

## 📞 Support

For questions or issues:

- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review the Jupyter notebook examples

---

**Note**: This is a research/academic project demonstrating ML techniques for financial market surveillance. Not intended for production use in real trading systems.
