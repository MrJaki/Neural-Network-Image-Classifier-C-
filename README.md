# Neural-Network-Image-Classifier-C-

This project is a simple feedforward neural network implemented entirely from scratch in C#.
It loads images, converts them to grayscale, normalizes pixel values, and trains a 16-neuron hidden layer to predict a 3-bit output.

The goal is to classify images into categories represented by binary vectors like:

0001

0110

1111
…etc.

The network supports saving/loading weights, testing custom images, and opening the selected image in Explorer.

✅ Features

No external ML libraries — everything coded manually.

Reads any bitmap image, converts to grayscale, and flattens it.

Trains using basic backpropagation.

Saves/loads model weights and biases (model.dat).

Lets you test with your own image path.

Opens test image after prediction.

Fully commented for learning purposes.

✅ Network Architecture

Input layer:
Size = number of grayscale pixels (depends on image resolution).

Hidden layer:
16 neurons
Activation: sigmoid

Output layer:
3 neurons
Activation: sigmoid
Output rounded to nearest bit (0/1).

✅ Usage
1. Add your training images

Inside the code:

string[] imagePaths = {
    // Add image paths here
};

2. Define matching expected outputs

The expected array must match the number of input images.

3. Run the program

You get three choices:

1 → Load saved model (no training)
2 → Load saved model AND continue training
anything else → Fresh training

4. Test with your own image

The program asks:

Would you like to use your own path for test picture?


If yes → paste the full path.

5. Model is saved automatically

After training, model.dat is created/updated.

✅ File Format (model.dat)

Binary layout:

length of weightsInputHidden

length of weightsOutputHidden

length of biasInput

length of biasOutput

all corresponding doubles

✅ Requirements

.NET 6 or newer

Path(s) to bitmap images
(Must be same resolution for training & testing or indexing breaks)
