package com.devankuleindiren.mnist;

import javax.swing.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.*;

/**
 * Created by Devan Kuleindiren on 21/09/15.
 */
public class CNN_28x28 extends SwingWorker<Double, Void> implements NeuralNetwork {

    // NUMBER OF NEURONS FOR FULLY CONNECTED LAYERS
    private int layer5NeuronNo = 50;
    private int layer6NeuronNo = 20;
    private int outputNeuronNo = 14;

    // THE DIMENSION OF THE FEATURE MAPS IN LAYERS C1 AND C2
    private int dimC1 = 24;
    private int dimC2 = 8;

    // KERNELS FOR THE CONVOLUTION LAYERS
    private Kernel[][] kernelsC1 = new Kernel[6][1];
    private Kernel[][] kernelsC2 = new Kernel[16][6];
    private Kernel[][] kernelsC3 = new Kernel[50][16];

    // WEIGHT MATRICES FOR FULLY CONNECTED LAYERS
    private Matrix weightsFC1 = new Matrix(layer5NeuronNo + 1, layer6NeuronNo);
    private Matrix weightsFC2 = new Matrix(layer6NeuronNo + 1, outputNeuronNo);

    // BOOLEAN ARRAYS THAT TRACK WHICH OF THE ACTIVATIONS FROM C1 AND C2 WERE MAX-POOLED
    private boolean[][][] C1Max = new boolean[6][dimC1][dimC1];
    private boolean[][][] C2Max = new boolean[16][dimC2][dimC2];

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
        for (int kernel = 0; kernel < kernelsC1.length; kernel++) {
            for (int input = 0; input < kernelsC1[kernel].length; input++) {
                kernelsC1[kernel][input] = new Kernel(5, 5);
                fillKernel(kernelsC1[kernel][input], 0.1);
            }
        }
        for (int kernel = 0; kernel < kernelsC2.length; kernel++) {
            for (int input = 0; input < kernelsC2[kernel].length; input++) {
                kernelsC2[kernel][input] = new Kernel(5, 5);
                fillKernel(kernelsC2[kernel][input], 0.1);
            }
        }
        for (int kernel = 0; kernel < kernelsC3.length; kernel++) {
            for (int input = 0; input < kernelsC3[kernel].length; input++) {
                kernelsC3[kernel][input] = new Kernel(4, 4);
                fillKernel(kernelsC3[kernel][input], 0.1);
            }
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

    // CHANGE THE FORMAT OF THE INPUT VECTORS TO ONE MORE CONVENIENT FOR THE CNN
    public Matrix[] preprocessTargetVectors (Matrix targetVectors) {
        Matrix[] targets = new Matrix[targetVectors.getHeight()];

        for (int vector = 0; vector < targetVectors.getHeight(); vector++) {
            targets[vector] = new Matrix(1, outputNeuronNo);
            for (int col = 0; col < outputNeuronNo; col++) {
                targets[vector].set(0, col, targetVectors.get(vector, col));
            }
        }
        return targets;
    }

    Matrix[][] inputs;
    Matrix[] targets;
    private double lR = 0.0001;
    private int iterationNo = 1;
    private Double error;

    @Override
    public void train(Matrix inputVectors, Matrix targets) throws MatrixDimensionMismatchException {
        this.inputs = preprocessInputVectors(inputVectors);
        this.targets = preprocessTargetVectors(targets);
        this.error = 0.0;

        final ControlPanel controlPanel = ControlPanel.getInstance();
        controlPanel.updateTrainingProgressBar(0, Strings.CONTROLPANEL_NEURALNETWORK_TRAININGINPROGRESS);

        this.addPropertyChangeListener(new PropertyChangeListener() {
            public  void propertyChange(PropertyChangeEvent evt) {
                if ("progress".equals(evt.getPropertyName())) {
                    controlPanel.updateTrainingProgressBar((Integer)evt.getNewValue(), Strings.CONTROLPANEL_NEURALNETWORK_TRAININGINPROGRESS);
                }
            }
        });

        this.execute();
    }

    @Override
    protected Double doInBackground() {

//        int batchSize = inputs.length;
        int batchSize = 2;

        try {
            for (int iteration = 0; iteration < iterationNo; iteration++) {

                error = 0.0;

                // STORE WEIGHT DELTAS
                Matrix dWFC2 = new Matrix(layer6NeuronNo + 1, outputNeuronNo);
                Matrix dWFC1 = new Matrix(layer5NeuronNo + 1, layer6NeuronNo);
                Kernel[][] dWC3 = new Kernel[50][16];
                Kernel[][] dWC2 = new Kernel[16][6];
                Kernel[][] dWC1 = new Kernel[6][1];

                // STORE PREVIOUS WEIGHT DELTAS
                Matrix dWFC2Prev = (Matrix) dWFC2.clone();
                Matrix dWFC1Prev = (Matrix) dWFC1.clone();
                Kernel[][] dWC3Prev = new Kernel[50][16];
                for (int k = 0; k < dWC3Prev.length; k++) {
                    for (int x = 0; x < dWC3Prev[k].length; x++) {
                        dWC3Prev[k][x] = (Kernel) dWC3[k][x].clone();
                    }
                }
                Kernel[][] dWC2Prev = new Kernel[16][6];
                for (int k = 0; k < dWC2Prev.length; k++) {
                    for (int x = 0; x < dWC2Prev[k].length; x++) {
                        dWC2Prev[k][x] = (Kernel) dWC2[k][x].clone();
                    }
                }
                Kernel[][] dWC1Prev = new Kernel[6][1];
                for (int k = 0; k < dWC1Prev.length; k++) {
                    for (int x = 0; x < dWC1Prev[k].length; x++) {
                        dWC1Prev[k][x] = (Kernel) dWC1[k][x].clone();
                    }
                }

                // INITIALISE THE ERROR GRADIENTS
                Matrix dEdWFC2 = new Matrix(layer6NeuronNo + 1, outputNeuronNo);
                Matrix dEdWFC1 = new Matrix(layer5NeuronNo + 1, layer6NeuronNo);
                Kernel[][] dEdWC3 = new Kernel[50][16];
                Kernel[][] dEdWC2 = new Kernel[16][6];
                Kernel[][] dEdWC1 = new Kernel[6][1];

                for (int currentInput = 0; currentInput < batchSize; currentInput++) {

                    // RESET THE MAX-POOL TRACKING ARRAYS
                    C1Max = new boolean[6][dimC1][dimC1];
                    C2Max = new boolean[16][dimC2][dimC2];

                    // FORWARD PASS
                    Matrix[] C1Activations = feedForwardConvolutionLayer(inputs[currentInput], kernelsC1);
                    Matrix[] S1Activations = feedForwardSubsampleLayer(C1Activations);
                    Matrix[] C2Activations = feedForwardConvolutionLayer(S1Activations, kernelsC2);
                    Matrix[] S2Activations = feedForwardSubsampleLayer(C2Activations);
                    Matrix[] C3Activations = feedForwardConvolutionLayer(S2Activations, kernelsC3);
                    Matrix FC1Input = flatten(C3Activations);
                    Matrix FC2Input = feedForwardFullyConnectedLayer1(FC1Input);
                    Matrix output = feedForwardFullyConnectedLayer2(FC2Input);

                    // REMOVE BIAS COLUMNS FOR TRAINING CALCULATIONS
                    Matrix FC2InputNoBias = FC2Input.removeBiasCol();
                    Matrix FC1InputNoBias = FC1Input.removeBiasCol();

                    // CALCULATE ERROR
                    for (int outputIndex = 0; outputIndex < output.getWidth(); outputIndex++) {
                        error += Math.pow((output.get(0, outputIndex) - targets[currentInput].get(0, outputIndex)), 2);
                    }

                    // PREPARE ONES MATRICES FOR CALCULATION
                    Matrix FC2Ones = Matrix.onesMatrix(1, outputNeuronNo);
                    Matrix FC1Ones = Matrix.onesMatrix(1, layer6NeuronNo);
                    Matrix C3Ones  = Matrix.onesMatrix(1, layer5NeuronNo);

                    // CALCULATE ALL DELTA TERMS
                    Matrix deltaFC2 = output.subtract(targets[currentInput]).multiplyEach(output.multiplyEach(FC2Ones.subtract(output)));
                    Matrix deltaFC1 = deltaFC2.multiply(weightsFC2.removeBiasRow().transpose()).multiplyEach(FC2InputNoBias.multiplyEach(FC1Ones.subtract(FC2InputNoBias)));
                    Matrix deltaC3  = deltaFC1.multiply(weightsFC1.removeBiasRow().transpose()).multiplyEach(FC1InputNoBias.multiplyEach(C3Ones.subtract(FC1InputNoBias)));
                    Matrix[] deltaS2 = new Matrix[16];
                    for (int x = 0; x < deltaS2.length; x++) {
                        deltaS2[x] = new Matrix(4, 4);
                        for (int row = 0; row < deltaS2[x].getHeight(); row++) {
                            for (int col = 0; col < deltaS2[x].getWidth(); col++) {
                                double newValue = 0;
                                for (int k = 0; k < deltaC3.getWidth(); k++) {
                                    newValue += deltaC3.get(0, k) * kernelsC3[k][x].get(row, col);
                                }
                                newValue = newValue * (S2Activations[x].get(row, col) * (1 - S2Activations[x].get(row, col)));
                                deltaS2[x].set(row, col, newValue);
                            }
                        }
                    }
                    Matrix[] deltaC2 = new Matrix[16];
                    for (int k = 0; k < deltaC2.length; k++) {
                        deltaC2[k] = new Matrix(8, 8);
                        for (int row = 0; row < deltaC2[k].getHeight(); row++) {
                            for (int col = 0; col < deltaC2[k].getWidth(); col++) {
                                if (C2Max[k][row][col]) deltaC2[k].set(row, col, deltaS2[k].get(row / 2, col / 2));
                            }
                        }
                    }
                    Matrix[] deltaS1 = new Matrix[6];
                    for (int x = 0; x < deltaS1.length; x++) {
                        deltaS1[x] = new Matrix(12, 12);
                        for (int row = 0; row < deltaS1[x].getHeight(); row++) {
                            for (int col = 0; col < deltaS1[x].getWidth(); col++) {
                                double newValue = 0;
                                for (int k = 0; k < deltaC2.length; k++) {
                                    for (int m = 0; m < dimC2; m++) {
                                        for (int n = 0; n < dimC2; n++) {
                                            if (m <= row
                                                    && row < m + dimC2
                                                    && n <= col
                                                    && col < n + dimC2) {
                                                newValue += deltaC2[k].get(m, n) * kernelsC2[k][x].get(m, n);
                                            }
                                        }
                                    }
                                }
                                newValue = newValue * (S1Activations[x].get(row, col) * (1 - S1Activations[x].get(row, col)));
                                deltaS1[x].set(row, col, newValue);
                            }
                        }
                    }
                    Matrix[] deltaC1 = new Matrix[6];
                    for (int k = 0; k < deltaC1.length; k++) {
                        deltaC1[k] = new Matrix(24, 24);
                        for (int row = 0; row < deltaC1[k].getHeight(); row++) {
                            for (int col = 0; col < deltaC1[k].getWidth(); col++) {
                                if (C1Max[k][row][col]) deltaC1[k].set(row, col, deltaS1[k].get(row / 2, col / 2));
                            }
                        }
                    }

                    // CALCULATE THE ERROR GRADIENTS
                    dEdWFC2 = dEdWFC2.add(FC2Input.transpose().multiply(deltaFC2));
                    dEdWFC1 = dEdWFC1.add(FC1Input.transpose().multiply(deltaFC1));
                    for (int k = 0; k < kernelsC3.length; k++) {
                        for (int x = 0; x < kernelsC3[k].length; x++) {
                            for (int row = 0; row < kernelsC3[k][x].getHeight(); row++) {
                                for (int col = 0; col < kernelsC3[k][x].getWidth(); col++) {
                                    dEdWC3[k][x].set(row, col, deltaC3.get(0, k) * S2Activations[x].get(row, col));
                                }
                            }
                        }
                    }
                    for (int k = 0; k < kernelsC2.length; k++) {
                        for (int x = 0; x < kernelsC2[k].length; x++) {
                            for (int row = 0; row < kernelsC2[k][x].getHeight(); row++) {
                                for (int col = 0; col < kernelsC2[k][x].getWidth(); col++) {
                                    double newValue = 0;
                                    for (int i = 0; i < dimC2; i++) {
                                        for (int j = 0; j < dimC2; j++) {
                                            newValue += deltaC2[k].get(i, j) * S1Activations[x].get(i + row, j + col);
                                        }
                                    }
                                    dEdWC2[k][x].set(row, col, newValue);
                                }
                            }
                        }
                    }
                    for (int k = 0; k < kernelsC1.length; k++) {
                        for (int x = 0; x < kernelsC1[k].length; x++) {
                            for (int row = 0; row < kernelsC1[k][x].getHeight(); row++) {
                                for (int col = 0; col < kernelsC1[k][x].getWidth(); col++) {
                                    double newValue = 0;
                                    for (int i = 0; i < dimC1; i++) {
                                        for (int j = 0; j < dimC1; j++) {
                                            newValue += deltaC1[k].get(i, j) * inputs[currentInput][x].get(i + row, j + col);
                                        }
                                    }
                                    dEdWC1[k][x].set(row, col, newValue);
                                }
                            }
                        }
                    }
                }

                // UPDATE WEIGHTS

                if (iteration % 1 == 0) System.out.println("Squared error: " + error);

                setProgress(100 * iteration / iterationNo);
            }
        } catch (MatrixDimensionMismatchException e) {
            JOptionPane.showMessageDialog(null, e.getMessage());
        }

        return error;
    }

    @Override
    protected void done () {
        ControlPanel controlPanel = ControlPanel.getInstance();
        controlPanel.updateTrainingProgressBar(100, Strings.CONTROLPANEL_NEURALNETWORK_TRAININGCOMPLETE);
        System.out.println("CNN trained with final squared error: " + error);
    }

    @Override
    public Matrix feedForward(Matrix inputVectors) throws MatrixDimensionMismatchException {
        Matrix[][] inputs = preprocessInputVectors(inputVectors);
        Matrix[] C1Activations = feedForwardConvolutionLayer(inputs[0], kernelsC1);
        Matrix[] S1Activations = feedForwardSubsampleLayer(C1Activations);
        Matrix[] C2Activations = feedForwardConvolutionLayer(S1Activations, kernelsC2);
        Matrix[] S2Activations = feedForwardSubsampleLayer(C2Activations);
        Matrix[] C3Activations = feedForwardConvolutionLayer(S2Activations, kernelsC3);
        Matrix FC1Input = flatten(C3Activations);
        Matrix FC2Input = feedForwardFullyConnectedLayer1(FC1Input);
        Matrix output = feedForwardFullyConnectedLayer2(FC2Input);

        return output;
    }

    // PERFORM A FOWARD PASS THROUGH A CONVOLUTIONAL LAYER IN WHICH KERNELS APPLY ALL INPUTS
    private Matrix[] feedForwardConvolutionLayer (Matrix[] inputs, Kernel[][] kernels) throws MatrixDimensionMismatchException {
        Matrix[] outputs = new Matrix[kernels.length];

        // APPLY CONVOLUTIONS
        for (int k = 0; k < kernels.length; k++) {
            Matrix output = new Matrix (inputs[0].getHeight() - kernels[0][0].getHeight() + 1, inputs[0].getWidth() - kernels[0][0].getWidth() + 1);
            for (int i = 0; i < inputs.length; i++) {
                output = output.add(kernels[k][i].convolute(inputs[i]));
            }
            outputs[k] = output;
        }

        for (Matrix o : outputs) o.applyLogisticActivation();

        return outputs;
    }

    // PERFORM A FORWARD PASS THROUGH A SUBSAMPLE LAYER USING MAX POOLING
    private Matrix[] feedForwardSubsampleLayer (Matrix[] inputs) {
        Matrix[] outputs = new Matrix[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = maxPool(inputs[i], i);
        }

        return outputs;
    }

    // METHOD FOR MAXPOOLING IN SUBSAMPLING LAYERS OF CNN
    private Matrix maxPool (Matrix input, int featureMapIndex) {
        Matrix result = new Matrix (input.getHeight() / 2, input.getWidth() / 2);

        boolean[][][] maxTrack;
        if (input.getHeight() == dimC1) maxTrack = C1Max;
        else maxTrack = C2Max;

        for (int row = 0; row < result.getHeight(); row++) {
            for (int col = 0; col < result.getWidth(); col++) {
                double max = input.get(row * 2, col * 2);
                int maxRow = row * 2;
                int maxCol = col * 2;

                if (input.get((row * 2) + 1, col * 2) > max) {
                    max = input.get((row * 2) + 1, col * 2);
                    maxRow = (row * 2) + 1;
                    maxCol = col * 2;
                }
                if (input.get(row * 2, (col * 2) + 1) > max) {
                    max = input.get(row * 2, (col * 2) + 1);
                    maxRow = row * 2;
                    maxCol = (col * 2) + 1;
                }
                if (input.get((row * 2) + 1, (col * 2) + 1) > max) {
                    max = input.get((row * 2) + 1, (col * 2) + 1);
                    maxRow = (row * 2) + 1;
                    maxCol = (col * 2) + 1;
                }
                result.set(row, col, max);
                maxTrack[featureMapIndex][maxRow][maxCol] = true;
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
        for (Kernel[] ks : kernelsC1) {
            for (Kernel k : ks) readKernel(k, bufferedReader);
        }

        // READ EACH OF THE C2 KERNELS
        for (Kernel[] ks : kernelsC2) {
            for (Kernel k : ks) readKernel(k, bufferedReader);
        }

        // READ EACH OF THE C3 KERNELS
        for (Kernel[] ks : kernelsC3) {
            for (Kernel k : ks) readKernel(k, bufferedReader);
        }

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
        for (Kernel[] ks : kernelsC1) {
            for (Kernel k : ks) writeKernel(k, bufferedWriter);
        }

        // WRITE EACH OF THE C2 KERNELS
        for (Kernel[] ks : kernelsC2) {
            for (Kernel k : ks) writeKernel(k, bufferedWriter);
        }

        // WRITE EACH OF THE C3 KERNELS
        for (Kernel[] ks : kernelsC3) {
            for (Kernel k : ks) writeKernel(k, bufferedWriter);
        }

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
