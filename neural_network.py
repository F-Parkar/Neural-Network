from import_data import get_data
import numpy as np
import matplotlib.pyplot as plt

# Constants for the network architecture
INPUT_LAYER_SIZE = 784  # Size of the input layer (28x28 pixels flattened)
HIDDEN_LAYER_SIZE = 20  # Size of the hidden layer
OUTPUT_LAYER_SIZE = 10  # Size of the output layer (10 classes for digits 0-9)

# Hyperparameters
LEARNING_RATE = 0.01
EPOCHS = 3

def initialize_weights_and_biases():
    """
    Initialize weights and biases for the input to hidden layer and hidden to output layer connections.
    """
    weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE))
    weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE))
    biases_hidden = np.zeros((HIDDEN_LAYER_SIZE, 1))
    biases_output = np.zeros((OUTPUT_LAYER_SIZE, 1))
    return weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output

def sigmoid(x):
    """
    Compute the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))

def forward_propagation(img, weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output):
    """
    Perform forward propagation through the network.
    """
    # Input to hidden layer
    hidden_layer_input = biases_hidden + weights_input_to_hidden @ img
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # Hidden to output layer
    output_layer_input = biases_output + weights_hidden_to_output @ hidden_layer_output
    output_layer_output = sigmoid(output_layer_input)
    
    return hidden_layer_output, output_layer_output

def compute_error(output, label):
    """
    Compute the mean squared error.
    """
    return 1 / len(output) * np.sum((output - label) ** 2, axis=0)

def backpropagation(img, label, hidden_layer_output, output_layer_output, weights_hidden_to_output, biases_output, biases_hidden, weights_input_to_hidden):
    """
    Perform backpropagation and update weights and biases.
    """
    # Output to hidden layer (cost function derivative)
    output_error = output_layer_output - label
    weights_hidden_to_output -= LEARNING_RATE * output_error @ hidden_layer_output.T
    biases_output -= LEARNING_RATE * output_error

    # Hidden to input layer (activation function derivative)
    hidden_error = weights_hidden_to_output.T @ output_error * (hidden_layer_output * (1 - hidden_layer_output))
    weights_input_to_hidden -= LEARNING_RATE * hidden_error @ img.T
    biases_hidden -= LEARNING_RATE * hidden_error

    return weights_hidden_to_output, biases_output, weights_input_to_hidden, biases_hidden

def train(images, labels):
    """
    Train the neural network using the training images and labels.
    """
    # Initialize weights and biases
    weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output = initialize_weights_and_biases()
    
    # Training loop for each epoch
    for epoch in range(EPOCHS):
        correct_predictions = 0
        
        # Training loop for each image and label
        for img, label in zip(images, labels):
            img.shape += (1,)  # Reshape image for matrix multiplication
            label.shape += (1,)  # Reshape label for matrix multiplication
            
            # Forward propagation
            hidden_layer_output, output_layer_output = forward_propagation(
                img, weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output
            )
            
            # Compute error and update correct predictions count
            error = compute_error(output_layer_output, label)
            correct_predictions += int(np.argmax(output_layer_output) == np.argmax(label))
            
            # Backpropagation and update weights and biases
            weights_hidden_to_output, biases_output, weights_input_to_hidden, biases_hidden = backpropagation(
                img, label, hidden_layer_output, output_layer_output, weights_hidden_to_output, biases_output, biases_hidden, weights_input_to_hidden
            )
        
        # Calculate and print accuracy for the current epoch
        accuracy = (correct_predictions / images.shape[0]) * 100
        print(f"Epoch {epoch + 1} Accuracy: {round(accuracy, 2)}%")
    
    return weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output

def predict(img, weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output):
    """
    Predict the class of a given image.
    """
    img.shape += (1,)  # Reshape image for matrix multiplication
    
    # Forward propagation
    hidden_layer_output, output_layer_output = forward_propagation(
        img, weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output
    )
    
    return output_layer_output.argmax()

def main():
    """
    Main function to train the network and make predictions.
    """
    # Load data
    images, labels = get_data()
    
    # Train the network
    weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output = train(images, labels)
    
    while True:
          # User input for image index
        user_input = input("Enter a number (0 - 59999) or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
        
        try:
            index = int(user_input)
            if index < 0 or index >= len(images):
                print("Index out of bounds. Please try again.")
                continue
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue
        
        img = images[index]
        
        # Display the image
        plt.imshow(img.reshape(28, 28), cmap="Greys")
        
        # Predict the class of the image
        prediction = predict(img, weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output)
        
        # Display the prediction
        plt.title(f"Prediction: {prediction}")
        plt.show()
        
if __name__ == "__main__":
    main()
