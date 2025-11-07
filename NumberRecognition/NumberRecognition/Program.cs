using System.Diagnostics;
using System.Drawing;
using System.Linq;

namespace NeuralNetwokMineV
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Load and normalize images
            string[] imagePaths = {
                // Add image paths here
            };

            List<double[]> data = new List<double[]>();
            foreach (var path in imagePaths)
            {
                if (!File.Exists(path))
                {
                    Console.WriteLine($"File not found: {path}");
                    continue;
                }

                using (Bitmap bitmap = new Bitmap(path))
                {
                    float[,] normalizedValues = GetNormalizedPixelValues(bitmap);
                    double[] flattenedValues = FlattenNormalizedValues(normalizedValues);
                    data.Add(flattenedValues);
                }
            }

            if (data.Count == 0)
            {
                Console.WriteLine("No valid image data loaded. Exiting.");
                return;
            }

            // Define expected outputs (replace with your desired outputs for training)
            double[][] expected = {
                new double[] {0, 0, 1}, // Corresponding to the first image
                new double[] {0, 0, 1},
                new double[] {0, 0, 1},
                new double[] {0, 1, 0},
                new double[] {0, 1, 0},
                new double[] {0, 1, 0},
                new double[] {0, 1, 1},
                new double[] {0, 1, 1},
                new double[] {0, 1, 1},
                new double[] {1, 0, 0},
                new double[] {1, 0, 0},
                new double[] {1, 0, 0},
                new double[] {1, 0, 1},
                new double[] {1, 0, 1},
                new double[] {1, 0, 1},
                new double[] {1, 1, 0},
                new double[] {1, 1, 0},
                new double[] {1, 1, 0},
                new double[] {1, 1, 1},
                new double[] {1, 1, 1},
                new double[] {1, 1, 1},
                // Add expected outputs for additional images
            };

            if (data.Count != expected.Length)
            {
                Console.WriteLine("Mismatch between input data and expected outputs. Exiting.");
                return;
            }

            // Defining sizes dynamically (input size depends on image resolution)
            // hiddenSize and outputSize are explicit and easy to change
            int inputSize = data[0].Length;     // will throw if data is empty (already checked above)
            int hiddenSize = 16;                // keep 16 or change as desired
            int outputSize = expected[0].Length;

            double[] weightsInputHidden = new double[hiddenSize * inputSize];
            double[] weightsOutputHidden = new double[outputSize * hiddenSize];
            double[] biasInput = new double[hiddenSize];
            double[] biasOutput = new double[outputSize];

            Random random = new Random();

            bool trainingDataSaved = false;
            Console.Write("Press 1 and submit if you want to use saved data \nPress 2 and submit if you want to use saved data for training \nPress any other key and submit to train model from beggining \n...  ");
            string loadData = Console.ReadLine();

            // Load model for later use
            if (File.Exists("model.dat") && loadData == "1")
            {
                LoadModel("model.dat", out weightsInputHidden, out weightsOutputHidden, out biasInput, out biasOutput);
                trainingDataSaved = true;
                Console.WriteLine("Data was saved...");
            } else if (File.Exists("model.dat") && loadData == "2")
            {
                LoadModel("model.dat", out weightsInputHidden, out weightsOutputHidden, out biasInput, out biasOutput);
                Console.WriteLine("Data was saved...");
            }
            else
            {
                Console.WriteLine("Data was not saved... creating weights and biases...");
                for (int i = 0; i < weightsInputHidden.Length; i++)
                    weightsInputHidden[i] = random.NextDouble() - 0.5;
                for (int i = 0; i < weightsOutputHidden.Length; i++)
                    weightsOutputHidden[i] = random.NextDouble() - 0.5;
                for (int i = 0; i < biasInput.Length; i++)
                    biasInput[i] = random.NextDouble() - 0.5; // center biases around 0
                for (int i = 0; i < biasOutput.Length; i++)
                    biasOutput[i] = random.NextDouble() - 0.5;
            }
            
            // Clycle settings
            int epoch = 2000; // Number of cycles for learning
            double learningRate = 0.1;

            // Training loop
            if (!trainingDataSaved)
            {
                for (int i = 0; i < epoch; i++)
                {
                    // Defining loop that will repeat itself data.Length times
                    for (int sample = 0; sample < data.Count; sample++)
                    {
                        double[] inputs = data[sample];
                        double[] hiddenOutputs = new double[hiddenSize];
                        double[] outputPredictions = new double[outputSize];

                        // Calculating for every neuron
                        for (int neuron = 0; neuron < hiddenOutputs.Length; neuron++)
                        {
                            double result = 0; // Seting up result

                            // Calculating neuron[i] to get every weights right in result
                            for (int input = 0; input < inputs.Length; input++)
                            {
                                result += inputs[input] * weightsInputHidden[neuron * inputs.Length + input];
                            }
                            result += biasInput[neuron];
                            hiddenOutputs[neuron] = Sigmoid(result);
                        }

                        // Calculating for every neuron
                        for (int neuron = 0; neuron < outputPredictions.Length; neuron++)
                        {
                            double result = 0;
                            // Calculating neuron[i] to get every weights right in result
                            for (int input = 0; input < hiddenOutputs.Length; input++)
                            {
                                result += hiddenOutputs[input] * weightsOutputHidden[neuron * hiddenOutputs.Length + input];

                            }
                            result += biasOutput[neuron];
                            outputPredictions[neuron] = Sigmoid(result);
                        }

                        // Defining how many errors there will be
                        // We can say that error is equal to output numer
                        double[] errorsOutput = new double[outputPredictions.Length];
                        for (int x = 0; x < outputPredictions.Length; x++)
                            errorsOutput[x] = expected[sample][x] - outputPredictions[x];

                        // Calculation of gradient output
                        double[] gradientsOutput = new double[outputPredictions.Length];
                        for (int x = 0; x < outputPredictions.Length; x++)
                            gradientsOutput[x] = errorsOutput[x] * SigmoidDerivative(outputPredictions[x]);

                        // Redefining weights and biases in hidden layer
                        for (int outputNeuron = 0; outputNeuron < outputPredictions.Length; outputNeuron++)
                        {
                            for (int hiddenNeuron = 0; hiddenNeuron < hiddenOutputs.Length; hiddenNeuron++)
                            {
                                weightsOutputHidden[outputNeuron * hiddenOutputs.Length + hiddenNeuron] += learningRate * gradientsOutput[outputNeuron] * hiddenOutputs[hiddenNeuron];
                            }
                            biasOutput[outputNeuron] += learningRate * gradientsOutput[outputNeuron];
                        }


                        // Calculation of errors on hidden layer
                        double[] errorHidden = new double[hiddenOutputs.Length];
                        for (int hiddenNeuron = 0; hiddenNeuron < hiddenOutputs.Length; hiddenNeuron++)
                        {
                            double errorResult = 0;
                            for (int outputNeuron = 0; outputNeuron < outputPredictions.Length; outputNeuron++)
                            {
                                errorResult += gradientsOutput[outputNeuron] * weightsOutputHidden[outputNeuron * hiddenOutputs.Length + hiddenNeuron];
                            }
                            errorHidden[hiddenNeuron] = errorResult * SigmoidDerivative(hiddenOutputs[hiddenNeuron]);
                        }

                        // Redefining weights and biases in input layer
                        for (int hiddenNeuron = 0; hiddenNeuron < hiddenOutputs.Length; hiddenNeuron++)
                        {
                            for (int input = 0; input < data[sample].Length; input++)
                            {
                                weightsInputHidden[hiddenNeuron * data[sample].Length + input] += learningRate * errorHidden[hiddenNeuron] * data[sample][input];
                            }
                            biasInput[hiddenNeuron] += learningRate * errorHidden[hiddenNeuron];
                        }
                    }

                    if (i % 20 == 0)
                    {
                        double completed = i / 20;
                        completed = Math.Round(completed);
                        int percentage = Convert.ToInt32(completed);
                        Console.WriteLine("{0:D2} %", percentage);
                    }
                }
            }

            List<string> testImages = new List<string>();

            Console.Write("Would you like to use your own path for test picture? ");
            string testSubject = Console.ReadLine();
            string imagePathInsert = null;

            if (testSubject.ToLower() == "yes")
            {
                Console.WriteLine("Insert image path: ");
                imagePathInsert = Console.ReadLine();
                testImages.Add(imagePathInsert);
            }
            else
            {
                // Add images using:
                // testImages.Add("");
            }

            List<double[]> testData = new List<double[]>();
            foreach (var path in testImages)
            {
                if (!File.Exists(path))
                {
                    Console.WriteLine($"File not found: {path}");
                    continue;
                }

                using (Bitmap bitmap = new Bitmap(path))
                {
                    float[,] normalizedValues = GetNormalizedPixelValues(bitmap);
                    double[] flattenedValues = FlattenNormalizedValues(normalizedValues);
                    testData.Add(flattenedValues);
                }
            }
            // Testing
            for (int n = 0; n < testData.Count; n++)
            {
                Console.WriteLine("Testing...");
                double[] testInput = testData[n];

                double[] testHiddenOutputs = new double[hiddenSize];
                double[] testOutputPredictions = new double[outputSize];

                // Hidden layer calculations
                for (int hiddenNeuron = 0; hiddenNeuron < testHiddenOutputs.Length; hiddenNeuron++)
                {
                    double sum = 0.0;
                    for (int input = 0; input < testInput.Length; input++)
                        sum += testInput[input] * weightsInputHidden[hiddenNeuron * testInput.Length + input];
                    sum += biasInput[hiddenNeuron];
                    testHiddenOutputs[hiddenNeuron] = Sigmoid(sum);
                }

                // Output layer calculations
                for (int outputNeuron = 0; outputNeuron < testOutputPredictions.Length; outputNeuron++)
                {
                    double sum = 0.0;
                    for (int hiddenNeuron = 0; hiddenNeuron < testHiddenOutputs.Length; hiddenNeuron++)
                        sum += testHiddenOutputs[hiddenNeuron] * weightsOutputHidden[outputNeuron * testHiddenOutputs.Length + hiddenNeuron];
                    sum += biasOutput[outputNeuron];
                    testOutputPredictions[outputNeuron] = Sigmoid(sum);
                }

                string resultBits = "";
                Console.WriteLine("Predicted Output for test input:");
                foreach (double output in testOutputPredictions)
                {
                    Console.WriteLine($"raw: {output:F6} -> bit: {Math.Round(output)}");
                    resultBits += (Math.Round(output) == 0 ? "0" : "1");
                }

                try
                {
                    long decimalValue = Convert.ToInt64(resultBits, 2);
                    Console.WriteLine($"Predicted number in binary -> {resultBits}  - decimal -> {decimalValue}");
                }
                catch
                {
                    Console.WriteLine($"Predicted bits: {resultBits} (could not convert to decimal)");
                }
                Console.WriteLine("------------------------------------");
            }

            string imagePath = @testImages[0]; // Replace with your image path
            OpenImageInGallery(imagePath);

            // Save model after training
            SaveModel(weightsInputHidden, weightsOutputHidden, biasInput, biasOutput, "model.dat");

            Console.ReadKey();

        }

        // Sigmoid function
        static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        // Derivative of sigmoid function
        static double SigmoidDerivative(double x)
        {
            return x * (1 - x);
        }

        // Load the model from a file
        static void LoadModel(string filename, out double[] weightsInputHidden, out double[] weightsOutputHidden, out double[] biasInput, out double[] biasOutput)
        {
            using (BinaryReader reader = new BinaryReader(File.Open(filename, FileMode.Open)))
            {
                int inputHiddenLength = reader.ReadInt32();
                int outputHiddenLength = reader.ReadInt32();
                int biasInputLength = reader.ReadInt32();
                int biasOutputLength = reader.ReadInt32();

                weightsInputHidden = new double[inputHiddenLength];
                weightsOutputHidden = new double[outputHiddenLength];
                biasInput = new double[biasInputLength];
                biasOutput = new double[biasOutputLength];

                for (int i = 0; i < inputHiddenLength; i++)
                    weightsInputHidden[i] = reader.ReadDouble();
                for (int i = 0; i < outputHiddenLength; i++)
                    weightsOutputHidden[i] = reader.ReadDouble();
                for (int i = 0; i < biasInputLength; i++)
                    biasInput[i] = reader.ReadDouble();
                for (int i = 0; i < biasOutputLength; i++)
                    biasOutput[i] = reader.ReadDouble();
            }
        }

        // Save the model to a file
        static void SaveModel(double[] weightsInputHidden, double[] weightsOutputHidden, double[] biasInput, double[] biasOutput, string filename)
        {
            using (BinaryWriter writer = new BinaryWriter(File.Open(filename, FileMode.Create)))
            {
                writer.Write(weightsInputHidden.Length);
                writer.Write(weightsOutputHidden.Length);
                writer.Write(biasInput.Length);
                writer.Write(biasOutput.Length);

                foreach (var weight in weightsInputHidden) writer.Write(weight);
                foreach (var weight in weightsOutputHidden) writer.Write(weight);
                foreach (var bias in biasInput) writer.Write(bias);
                foreach (var bias in biasOutput) writer.Write(bias);
            }
        }

        static float[,] GetNormalizedPixelValues(Bitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;
            float[,] normalizedValues = new float[width, height];

            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    Color pixel = bitmap.GetPixel(x, y);
                    float grayscale = (pixel.R + pixel.G + pixel.B) / (3.0f * 255.0f);
                    normalizedValues[x, y] = grayscale;
                }
            }

            return normalizedValues;
        }

        static double[] FlattenNormalizedValues(float[,] values)
        {
            int width = values.GetLength(0);
            int height = values.GetLength(1);
            double[] flattened = new double[width * height];
            int index = 0;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    flattened[index++] = values[x, y];
                }
            }

            return flattened;
        }

        static void OpenImageInGallery(string imagePath)
        {
            try
            {
                Process.Start("explorer", imagePath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred: {ex.Message}");
            }
        }
    }
}
