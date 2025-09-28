# ModelFlex: ML Model Optimization Tool

ModelFlex is a powerful web-based tool for optimizing machine learning models across different frameworks and target devices. It provides an intuitive interface for model compression, optimization, and deployment.

## Features

- **Multi-Framework Support**
  - TensorFlow (.h5, .pb)
  - PyTorch (.pt, .pth)
  - ONNX (.onnx)

- **Target Device Optimization**
  - CPU optimization with quantization
  - GPU acceleration
  - TPU compatibility (TensorFlow models)

- **Optimization Techniques**
  - Model quantization (INT8, FP16)
  - Operation fusion
  - Graph optimization
  - Memory optimization
  - Platform-specific optimizations

- **Performance Metrics**
  - Model size reduction
  - Inference latency
  - Detailed comparison reports

## Tech Stack

### Frontend
- React with Vite
- TailwindCSS for styling
- Modern component architecture
- Real-time progress tracking

### Backend
- FastAPI (Python)
- Async processing
- Multiple ML framework support
- Efficient file handling

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 16+
- NPM or Yarn
- CUDA toolkit (optional, for GPU support)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ModelFlex.git
cd ModelFlex
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

### Running the Application

1. Start the backend server:
```bash
cd backend
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
python -m uvicorn app.main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

3. Open your browser and navigate to `http://localhost:5173`

## Usage

1. Upload your ML model (.h5, .pb, .pt, .pth, or .onnx format)
2. Select your target device (CPU, GPU, or TPU)
3. Click "Optimize Model"
4. View optimization results and download the optimized model

## Optimization Details

### TensorFlow Models
- CPU: FP16 quantization, operation fusion
- GPU: GPU-specific optimizations, FP16 support
- TPU: INT8 quantization with fallback options

### PyTorch Models
- CPU: Dynamic quantization, operation fusion
- GPU: TorchScript compilation, CUDA optimizations

### ONNX Models
- CPU: Multi-threading, memory optimizations
- GPU: CUDA execution provider, graph optimizations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow team for TFLite conversion tools
- PyTorch team for quantization and TorchScript
- ONNX Runtime team for optimization capabilities
- FastAPI team for the excellent web framework