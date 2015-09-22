package com.devankuleindiren.mnist;

import javax.swing.*;
import java.io.*;

/**
 * Created by Devan Kuleindiren on 21/09/15.
 */
public class CNN_28x28 extends SwingWorker<Double, Void> implements NeuralNetwork {

    // THIS CONVOLUTIONAL NEURAL NETWORK HAS A FIXED STRUCTURE
    final private static int inputNodesNo = 785;

    // KERNELS FOR THE CONVOLUTION LAYERS
    private Kernel[] kernelsC1 = new Kernel[6];
    private Kernel[] kernelsC2 = new Kernel[16];
    private Kernel[] kernelsC3 = new Kernel[50];

    // NUMBER OF NEURONS FOR FULLY CONNECTED LAYERS
    private int layer5NeuronNo = 50;
    private int layer6NeuronNo = 20;
    private int outputNeuronNo = 14;

    // WEIGHT MATRICES FOR FULLY CONNECTED LAYERS
    private Matrix weightsFC1 = new Matrix(layer5NeuronNo + 1, layer6NeuronNo);
    private Matrix weightsFC2 = new Matrix(layer6NeuronNo + 1, outputNeuronNo);

    public CNN_28x28 () {
        initWeights();
        initKernels();
    }

    // INITIALISE EACH WEIGHT TO A RANDOM VALUE, X, IN THE RANGE -N^(-0.5) < X < N^(-0.5)
    // WHERE N IS THE NUMBER OF NODES IN THE LAYER BEFORE THE WEIGHTS
    private void fillRandom (Matrix matrix, int noOfInputs) {
        for (int row = 0; row < matrix.getHeight(); row++) {
            for (int col = 0; col < matrix.getWidth(); col++) {
                matrix.set(row, col, (Math.random() * (2*(1/Math.pow(noOfInputs, 0.5)))) - (1/Math.pow(noOfInputs, 0.5)));
            }
        }
    }

    // INITIALISE THE WEIGHT MATRICES
    private void initWeights() {
        fillRandom(weightsFC1, layer5NeuronNo);
        fillRandom(weightsFC2, layer6NeuronNo);
    }

    // INITIALISE EACH KERNEL VALUE TO A RANDOM VALUE BETWEEN +bound AND -bound
    private void fillKernel (Kernel kernel, double bound) {
        for (int row = 0; row < kernel.getHeight(); row++) {
            for (int col = 0; col < kernel.getWidth(); col++) {
                kernel.set(row, col, (Math.random() * 2 * bound) - bound);
            }
        }
        kernel.setBiasWeight((Math.random() * 2 * bound) - bound);
    }

    // INITIALISE THE KERNELS
    private void initKernels () {
        for (int k = 0; k < kernelsC1.length; k++) {
            kernelsC1[k] = new Kernel(5, 5);
            fillKernel(kernelsC1[k], 0.1);
        }
        for (int k = 0; k < kernelsC2.length; k++) {
            kernelsC2[k] = new Kernel(5, 5);
            fillKernel(kernelsC2[k], 0.1);
        }
        for (int k = 0; k < kernelsC3.length; k++) {
            kernelsC3[k] = new Kernel(4, 4);
            fillKernel(kernelsC3[k], 0.1);
        }
    }

    // CHANGE THE FORMAT OF THE INPUT VECTORS TO ONE MORE CONVENIENT FOR THE CNN
    public Matrix[][] preprocessInputVectors (Matrix inputVectors) {
        Matrix[][] inputs = new Matrix[inputVectors.getHeight()][1];

        for (int vector = 0; vector < inputVectors.getHeight(); vector++) {
            inputs[vector][0] = new Matrix(28, 28);
            for (int row = 0; row < 28; row++) {
                for (int col = 0; col < 28; col++) {
                    inputs[vector][0].set(row, col, inputVectors.get(vector, (row * 28) + col));
                }
            }
        }
        return inputs;
    }

    @Override
    protected Double doInBackground() throws Exception {
        return null;
    }

    @Override
    protected void done () {
        ControlPanel controlPanel = ControlPanel.getInstance();
        controlPanel.updateTrainingProgressBar(100, Strings.CONTROLPANEL_NEURALNETWORK_TRAININGCOMPLETE);
    }

    @Override
    public double train(Matrix inputVectors, Matrix targets) throws MatrixDimensionMismatchException {
        return 0;
    }

    @Override
    public Matrix feedForward(Matrix inputVectors) throws MatrixDimensionMismatchException {
        Matrix[][] inputs = preprocessInputVectors(inputVectors);
        Matrix[] C1Activations = feedForwardConvolutionLayer(inputs[0], kernelsC1);
        Matrix[] S1Activations = feedForwardSubsampleLayer(C1Activations);
        // EACH ROW DETERMINES WHICH INPUTS A KERNEL APPLIES TO
        boolean[][] kernelConnections = {
                {true, true, true, false, false, false},
                {false, true, true, true, false, false},
                {false, false, true, true, true, false},
                {false, false, false, true, true, true},
                {true, false, false, false, true, true},
                {true, true, false, false, false, true},
                {true, true, true, true, false, false},
                {false, true, true, true, true, false},
                {false, false, true, true, true, true},
                {true, false, false, true, true, true},
                {true, true, false, false, true, true},
                {true, true, true, false, false, true},
                {true, true, false, true, true, false},
                {false, true, true, false, true, true},
                {true, false, true, true, false, true},
                {true, true, true, true, true, true},
        };
        Matrix[] C2Activations = feedForwardConvolutionLayer(S1Activations, kernelsC2, kernelConnections);
        Matrix[] S2Activations = feedForwardSubsampleLayer(C2Activations);
        Matrix[] C3Activations = feedForwardConvolutionLayer(S2Activations, kernelsC3);
        Matrix FC1Input = flatten(C3Activations);
        Matrix FC2Input = feedForwardFullyConnectedLayer1(FC1Input);
        Matrix output = feedForwardFullyConnectedLayer2(FC2Input);

        return output;
    }

    // PERFORM A FOWARD PASS THROUGH A CONVOLUTIONAL LAYER IN WHICH KERNELS APPLY ALL INPUTS
    private Matrix[] feedForwardConvolutionLayer (Matrix[] inputs, Kernel[] kernels) throws MatrixDimensionMismatchException {
        Matrix[] outputs = new Matrix[kernels.length];

        // APPLY CONVOLUTIONS
        for (int k = 0; k < kernels.length; k++) {
            Matrix output = new Matrix (inputs[0].getHeight() - kernels[0].getHeight() + 1, inputs[0].getWidth() - kernels[0].getWidth() + 1);
            for (int i = 0; i < inputs.length; i++) {
                output.add(kernels[k].convolute(inputs[i]));
            }
            outputs[k] = output;
        }

        return outputs;
    }

    // PERFORM A FOWARD PASS THROUGH A CONVOLUTIONAL LAYER IN WHICH KERNELS APPLY TO SPECIFIC INPUTS
    private Matrix[] feedForwardConvolutionLayer (Matrix[] inputs, Kernel[] kernels, boolean[][] kernelConnections) throws MatrixDimensionMismatchException {
        Matrix[] outputs = new Matrix[kernels.length];

        // APPLY CONVOLUTIONS
        for (int k = 0; k < kernels.length; k++) {
            Matrix output = new Matrix (inputs[0].getHeight() - kernels[0].getHeight() + 1, inputs[0].getWidth() - kernels[0].getWidth() + 1);
            for (int i = 0; i < kernelConnections[k].length; i++) {
                if (kernelConnections[k][i]) {
                    output.add(kernels[k].convolute(inputs[i]));
                }
            }
            outputs[k] = output;
        }

        return outputs;
    }

    // PERFORM A FORWARD PASS THROUGH A SUBSAMPLE LAYER USING MAX POOLING
    private Matrix[] feedForwardSubsampleLayer (Matrix[] inputs) {
        Matrix[] outputs = new Matrix[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = maxPool(inputs[i]);
        }

        return outputs;
    }

    // METHOD FOR MAXPOOLING IN SUBSAMPLING LAYERS OF CNN
    private Matrix maxPool (Matrix input) {
        Matrix result = new Matrix (input.getHeight() / 2, input.getWidth() / 2);

        for (int row = 0; row < result.getHeight(); row++) {
            for (int col = 0; col < result.getWidth(); col++) {
                double max = input.get(row * 2, col * 2);

                if (input.get((row * 2) + 1, col * 2) > max) max = input.get((row * 2) + 1, col * 2);
                if (input.get(row * 2, (col * 2) + 1) > max) max = input.get(row * 2, (col * 2) + 1);
                if (input.get((row * 2) + 1, (col * 2) + 1) > max) max = input.get((row * 2) + 1, (col * 2) + 1);

                result.set(row, col, max);
            }
        }

        return result;
    }

    // FLATTEN OUTPUTS OF THE C3 LAYER INTO ONE LONG INPUT VECTOR FOR THE FULLY CONNECTED LAYERS
    // THIS ASSUMES EACH OF THE MATRICES IN matrices HAS DIMENSION 1x1
    private Matrix flatten (Matrix[] matrices) {
        Matrix result = new Matrix(1, matrices.length + 1);

        for (int col = 0; col < result.getWidth() - 1; col++) {
            result.set(0, col, matrices[col].get(0, 0));
        }

        // ADD THE BIAS INPUT
        result.set(0, result.getWidth() - 1, -1);

        return result;
    }

    private Matrix feedForwardFullyConnectedLayer1 (Matrix input) throws MatrixDimensionMismatchException {
        return input.feedForwardAndAddBias(weightsFC1);
    }

    private Matrix feedForwardFullyConnectedLayer2 (Matrix input) throws MatrixDimensionMismatchException {
        Matrix output = input.multiply(weightsFC2);
        output.applyLogisticActivation();

        return output;
    }

    @Override
    public void loadWeights(String source) throws FileNotFoundException, IOException, InvalidWeightFormatException {
        BufferedReader bufferedReader = new BufferedReader(new FileReader(source));

        // READ EACH OF THE C1 KERNELS
        for (Kernel k : kernelsC1) readKernel(k, bufferedReader);

        // READ EACH OF THE C2 KERNELS
        for (Kernel k : kernelsC2) readKernel(k, bufferedReader);

        // READ EACH OF THE C3 KERNELS
        for (Kernel k : kernelsC3) readKernel(k, bufferedReader);

        // READ THE FC1 WEIGHTS
        readWeights(weightsFC1, bufferedReader);

        // READ THE FC2 WEIGHTS
        readWeights(weightsFC2, bufferedReader);

        bufferedReader.close();
        JOptionPane.showMessageDialog(null, "Loaded weights from " + source);
    }

    @Override
    public void saveWeights(String destination) throws FileNotFoundException, IOException {
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(destination, false));

        // WRITE EACH OF THE C1 KERNELS
        for (Kernel k : kernelsC1) writeKernel(k, bufferedWriter);

        // WRITE EACH OF THE C2 KERNELS
        for (Kernel k : kernelsC2) writeKernel(k, bufferedWriter);

        // WRITE EACH OF THE C3 KERNELS
        for (Kernel k : kernelsC3) writeKernel(k, bufferedWriter);

        // WRITE THE FC1 WEIGHTS
        writeWeights(weightsFC1, bufferedWriter);
        bufferedWriter.write("\n");

        // WRITE THE FC2 WEIGHTS
        writeWeights(weightsFC2, bufferedWriter);

        bufferedWriter.close();
        JOptionPane.showMessageDialog(null, "Saved weights to " + destination);
    }

    private void writeKernel (Kernel kernel, BufferedWriter bufferedWriter) throws IOException {
        for (int row = 0; row < kernel.getHeight(); row++) {
            for (int col = 0; col < kernel.getWidth(); col++) {
                bufferedWriter.write(Double.toString(kernel.get(row, col)) + ",");
            }
        }
        bufferedWriter.write(Double.toString(kernel.getBiasWeight()));
        bufferedWriter.write("\n");
    }

    private void readKernel (Kernel kernel, BufferedReader bufferedReader) throws IOException, InvalidWeightFormatException {
        try {
            String kernelString = bufferedReader.readLine();
            String[] kernelArray = kernelString.split(",");
            for (int row = 0; row < kernel.getHeight(); row++) {
                for (int col = 0; col < kernel.getWidth(); col++) {
                    kernel.set(row, col, Double.parseDouble(kernelArray[(row * kernel.getWidth()) + col]));
                }
            }
            kernel.setBiasWeight(Double.parseDouble(kernelArray[kernel.getHeight() * kernel.getWidth()]));
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new InvalidWeightFormatException("A kernel in the file is incorrectly formatted.");
        }
    }

    private void writeWeights (Matrix weights, BufferedWriter bufferedWriter) throws IOException {
        for (int row = 0; row < weights.getHeight(); row++) {
            for (int col = 0; col < weights.getWidth(); col++) {
                bufferedWriter.write(Double.toString(weights.get(row, col)) + ",");
            }
        }
    }

    private void readWeights (Matrix weights, BufferedReader bufferedReader) throws IOException, InvalidWeightFormatException {
        try {
            String weightString = bufferedReader.readLine();
            String[] weightArray = weightString.split(",");
            for (int row = 0; row < weights.getHeight(); row++) {
                for (int col = 0; col < weights.getWidth(); col++) {
                    weights.set(row, col, Double.parseDouble(weightArray[(row * weights.getWidth()) + col]));
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new InvalidWeightFormatException("A weight matrix in the file is incorrectly formatted.");
        }
    }
}
