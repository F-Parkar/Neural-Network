# MNIST Neural Network

This project implements a simple neural network to classify handwritten digits from the MNIST dataset. The network is built from scratch using NumPy and trained using forward and backpropagation.

## Project Structure
├── import_data.py

├── neural_network.py

└── README.md


- `import_data.py`: Contains the `get_data()` function to load the MNIST dataset.
- `neural_network.py`: Main script to define, train, and test the neural network.

## Getting Started

### Installing
1. Clone the repository:
   git clone https://github.com/your-username/MNIST-Neural-Network.git
   cd MNIST-Neural-Network

2. Create a virtual environment and activate it:
   python -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`

3. Install the requirements.txt:
   pip install -r requirements.txt

## Running the Code
Make sure you have the MNIST dataset. You can modify import_data.py to download and preprocess the data.
Run the neural network script: python neural_network.py

## Usage
The script will train the neural network on the MNIST dataset for a specified number of epochs.
After training, you can test the network by inputting an index (0 - 59999) to see the prediction for that image. Enter 'q' to quit.

## Functions
`initialize_weights_and_biases()`
Initializes weights and biases for the input to hidden and hidden to output layer connections.

`sigmoid(x)`
Computes the sigmoid activation function.

`forward_propagation(img, weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output)`
Performs forward propagation through the network.

`compute_error(output, label`
Computes the mean squared error.

`backpropagation(img, label, hidden_layer_output, output_layer_output, weights_hidden_to_output, biases_output, biases_hidden, weights_input_to_hidden)`
Performs backpropagation and updates weights and biases.

`train(images, labels)`
Trains the neural network using the training images and labels.

`predict(img, weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output)`
Predicts the class of a given image.

## Acknowledgments
This project was inspired from the youtube channel: https://www.youtube.com/@BotAcademyYT
   
   



