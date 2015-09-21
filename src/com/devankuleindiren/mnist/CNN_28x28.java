package com.devankuleindiren.mnist;

import javax.swing.*;
import java.io.FileNotFoundException;
import java.io.IOException;

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
    private Matrix weights1 = new Matrix(layer5NeuronNo + 1, layer6NeuronNo);
    private Matrix weights2 = new Matrix(layer6NeuronNo + 1, outputNeuronNo);

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
        fillRandom(weights1, layer5NeuronNo);
        fillRandom(weights2, layer6NeuronNo);
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

        // TODO: Flatten C3Activations and pass it through the fully connected layers as normal

        return null;
    }

    // PERFORM A FOWARD PASS THROUGH A CONVOLUTIONAL LAYER IN WHICH KERNELS APPLY ALL INPUTS
    private Matrix[] feedForwardConvolutionLayer (Matrix[] inputs, Kernel[] kernels) throws MatrixDimensionMismatchException {
        Matrix[] outputs = new Matrix[kernels.length];

        // APPLY CONVOLUTIONS
        for (int k = 0; k < kernels.length; k++) {
            Matrix output = new Matrix (inputs[0].getHeight() - kernels[0].getHeight() + 1, inputs[0].getWidth() - kernels[0].getWidth() + 1);
            for (int i = 0; i < inputs.length; i++) {
                output.add(kernels[k].apply(inputs[i]));
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
                    output.add(kernels[k].apply(inputs[i]));
                }
            }
            outputs[k] = output;
        }

        return outputs;
    }

    private Matrix[] feedForwardSubsampleLayer (Matrix[] inputs) {
        Matrix[] outputs = new Matrix[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = maxPool(inputs[i]);
        }

        return outputs;
    }

    // METHOD FOR MAXPOOLING IN SUBSAMPLING LAYERS OF CNN
    public Matrix maxPool (Matrix input) {
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

    @Override
    public void loadWeights(String source) throws FileNotFoundException, IOException, InvalidWeightFormatException {

    }

    @Override
    public void saveWeights(String destination) throws FileNotFoundException, IOException {

    }
}
