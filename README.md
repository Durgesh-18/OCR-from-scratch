# OCR Neural Network Demo

A browser-based optical character recognition (OCR) system that uses a simple neural network to recognize hand-drawn digits (0-9). Draw digits on a canvas, train the network, and test predictions in real-time.

## Features

- **Interactive Canvas**: Draw digits using your mouse on a 20x20 pixel grid
- **Real-time Training**: Train the neural network directly from the browser
- **Instant Predictions**: Test the network's ability to recognize your drawings
- **Persistent Learning**: Network weights are saved to disk and loaded on restart

## Architecture

The system consists of three main components:

### 1. Frontend (`ocr.html`, `ocr.js`)
- HTML5 canvas for drawing digits
- JavaScript interface for user interactions
- Sends training data and prediction requests to the backend

### 2. Backend Server (`server.py`)
- HTTP server running on port 8000
- Handles training and prediction requests
- Manages neural network persistence

### 3. Neural Network (`ocr.py`)
- 3-layer feedforward neural network
- Input layer: 400 nodes (20x20 pixel grid)
- Hidden layer: 25 nodes (configurable)
- Output layer: 10 nodes (digits 0-9)
- Backpropagation with sigmoid activation

## Installation

### Prerequisites
- Python 3.x
- NumPy

### Setup

1. Install dependencies:
```bash
pip install numpy
```

2. Clone or download all project files to a directory:
   - `ocr.html`
   - `ocr.js`
   - `ocr.py`
   - `server.py`
   - `neural_network_design.py` (optional, for testing)

## Usage

### 1. Start the Server

```bash
python server.py
```

You should see:
```
Server running on port 8000
```

### 2. Open the Interface

Open `ocr.html` in your web browser (double-click the file or open it via File > Open).

### 3. Train the Network

1. Draw a digit (0-9) on the canvas by clicking and dragging
2. Enter the digit value in the "Digit" input field
3. Click "Train"
4. Repeat 10 times (the network trains in batches of 10)

### 4. Test Predictions

1. Draw a digit on the canvas
2. Click "Test"
3. See the predicted digit in an alert box

### 5. Reset Canvas

Click "Reset" to clear the canvas and draw again.

## Configuration

### Neural Network Parameters

In `ocr.py`, you can modify:

```python
# Number of hidden layer nodes (default: 25)
nn = OCRNeuralNetwork(num_hidden_nodes=25)

# Learning rate (default: 0.1)
LEARNING_RATE = 0.1
```

### Server Settings

In `server.py`:

```python
PORT = 8000  # Change server port
```

In `ocr.js`:

```python
HOST: "http://localhost",  # Server address
PORT: 8000,                # Server port
BATCH_SIZE: 10,           # Training batch size
```

### Canvas Settings

In `ocr.js`:

```javascript
CANVAS_WIDTH: 200,      // Canvas pixel width
TRANSLATED_WIDTH: 20,   // Grid resolution (20x20)
PIXEL_WIDTH: 10,        // Individual pixel size
```

## Testing Performance

Use `neural_network_design.py` to test different network configurations:

```python
# Tests hidden layer sizes from 5 to 50 nodes
for i in xrange(5, 50, 5):
    nn = OCRNeuralNetwork(i, data_matrix, data_labels, train_indices, False)
    performance = str(test(data_matrix, data_labels, test_indices, nn))
    print "{i} Hidden Nodes: {val}".format(i=i, val=performance)
```

This requires a labeled dataset in the appropriate format.

## How It Works

### Training Process

1. User draws a digit and labels it
2. After 10 drawings, data is sent to the server
3. Server performs forward propagation:
   - Input → Hidden layer (with sigmoid activation)
   - Hidden → Output layer (with sigmoid activation)
4. Backpropagation calculates errors and updates weights
5. Updated weights are saved to `nn.json`

### Prediction Process

1. User draws a digit
2. Pixel data is sent to the server
3. Server performs forward propagation
4. Returns the digit with the highest output activation (argmax)

### Network Equations

**Forward Propagation:**
```
y1 = sigmoid(θ1 · x + b1)
y2 = sigmoid(θ2 · y1 + b2)
```

**Backpropagation:**
```
output_error = actual - y2
hidden_error = θ2^T · output_error ⊙ sigmoid'(z1)

θ2 += learning_rate · output_error · y1^T
θ1 += learning_rate · hidden_error · x^T
```

## File Structure

```
.
├── ocr.html                    # Frontend interface
├── ocr.js                      # Frontend logic
├── ocr.py                      # Neural network implementation
├── server.py                   # HTTP server
├── neural_network_design.py    # Performance testing (optional)
├── nn.json                     # Saved network weights (generated)
└── README.md                   # This file
```

## Troubleshooting

### "Network error: server unreachable"
- Ensure `server.py` is running
- Check that the port (8000) is not blocked by a firewall
- Verify the HOST and PORT settings in `ocr.js` match your server

### Poor Prediction Accuracy
- Train with more examples (50+ per digit recommended)
- Try different numbers of hidden nodes
- Ensure drawings are clear and centered
- Reset and retrain if the network becomes overtrained on bad data

### CORS Issues
- The server includes CORS headers for cross-origin requests
- If issues persist, serve `ocr.html` from the same origin as the server

## Technical Details

- **Input Format**: 400-dimensional vector (flattened 20x20 grid, values 0 or 1)
- **Weight Initialization**: Random values between -0.06 and 0.06
- **Activation Function**: Sigmoid (logistic function)
- **Training Algorithm**: Batch gradient descent with backpropagation
- **Persistence**: JSON format for easy debugging and portability

## Limitations

- Only recognizes digits 0-9
- Requires manual training for each session (unless using saved weights)
- Performance depends on drawing style and training quality
- No data augmentation or preprocessing

## Future Improvements

- Add data augmentation (rotation, scaling, translation)
- Implement mini-batch or stochastic gradient descent
- Add validation set and early stopping
- Support for letters and symbols
- Pre-trained weights with MNIST dataset
- Real-time accuracy metrics display

## License

This is an educational project. Feel free to use and modify as needed.

## Credits

Built as a demonstration of basic neural network concepts using vanilla JavaScript and Python with NumPy.
