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

    // WEIGHT MATRICES FOR FULLY CONNECTED LAYERS
    private Matrix weights1 = new Matrix(51, 20);
    private Matrix weights2 = new Matrix(21, 14);

    public CNN_28x28 () {

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
    public Matrix feedForward(Matrix inputVectors) throws MatrixDimensionMismatchException {
        return null;
    }

    @Override
    public void loadWeights(String source) throws FileNotFoundException, IOException, InvalidWeightFormatException {

    }

    @Override
    public void saveWeights(String destination) throws FileNotFoundException, IOException {

    }
}
