package com.devankuleindiren.mnist;

import javax.swing.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.*;

/**
 * Created by Devan Kuleindiren on 29/08/15.
 */

public class FNN2Layer implements NeuralNetwork {

    public FNN2Layer (int iNN, int hNN, int oNN) {
        inputNodesNo = iNN;
        hiddenNeuronNo = hNN;
        outputNeuronNo = oNN;

        weights1 = new Matrix(inputNodesNo, hiddenNeuronNo);
        weights2 = new Matrix(hiddenNeuronNo + 1, outputNeuronNo);

        initWeights();
    }

    // SPECIFY NUMBER OF NODES IN INPUT, HIDDEN AND OUTPUT LAYERS
    private int inputNodesNo;
    private int hiddenNeuronNo;
    private int outputNeuronNo;

    // WEIGHT MATRICES
    private Matrix weights1;
    private Matrix weights2;

    public void setWeights (Matrix weights1, Matrix weights2) {
        this.weights1 = weights1;
        this.weights2 = weights2;
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
        fillRandom(weights1, inputNodesNo);
        fillRandom(weights2, hiddenNeuronNo);
    }

    // TRAIN THE NET USING GIVEN INPUTS & TARGETS, WITH A GIVEN LEARNING RATE AND NUMBER OF ITERATIONS
    @Override
    public void train (Matrix inputVectors, Matrix targets) throws MatrixDimensionMismatchException {

        final ControlPanel controlPanel = ControlPanel.getInstance();
        controlPanel.updateTrainingProgressBar(0, Strings.CONTROLPANEL_NEURALNETWORK_TRAININGINPROGRESS);

        FNN2Layer_Train fnn2Layer_train = new FNN2Layer_Train(this, weights1, weights2, hiddenNeuronNo, outputNeuronNo, inputVectors, targets);

        fnn2Layer_train.addPropertyChangeListener(new PropertyChangeListener() {
            public  void propertyChange(PropertyChangeEvent evt) {
                if ("progress".equals(evt.getPropertyName())) {
                    controlPanel.updateTrainingProgressBar((Integer)evt.getNewValue(), Strings.CONTROLPANEL_NEURALNETWORK_TRAININGINPROGRESS);
                }
            }
        });

        fnn2Layer_train.execute();
    }

    // PASS AN ARRAY OF INPUT VECTORS THROUGH THE NEURAL NET
    @Override
    public Matrix feedForward (Matrix inputVectors) throws MatrixDimensionMismatchException {
        Matrix hiddenActs;
        Matrix outputActs;

        // INPUT NODES TO HIDDEN LAYER
        hiddenActs = feedForwardP1(inputVectors);

        // HIDDEN LAYER TO OUTPUT LAYER
        outputActs = feedForwardP2(hiddenActs);

        return outputActs;
    }

    // COMPUTE THE ACTIVATIONS OF THE HIDDEN LAYER NODES GIVEN THE INPUT VECTORS
    private Matrix feedForwardP1 (Matrix inputVectors) throws MatrixDimensionMismatchException {
        return inputVectors.feedForwardAndAddBias(weights1);
    }

    // COMPUTE THE ACTIVATIONS OF THE OUTPUT LAYER NODES GIVEN THE HIDDEN LAYER ACTIVATIONS
    private Matrix feedForwardP2 (Matrix hiddenActs) throws MatrixDimensionMismatchException {
        Matrix outputActs;

        outputActs = hiddenActs.multiply(weights2);
        outputActs.applyLogisticActivation();

        return outputActs;
    }

    // LOAD WEIGHTS FROM GIVEN SOURCE FILE
    @Override
    public void loadWeights (String source) throws FileNotFoundException, IOException, InvalidWeightFormatException {
        BufferedReader bufferedReader = new BufferedReader(new FileReader(source));

        String metadataLine = bufferedReader.readLine();
        String[] metadata = metadataLine.split(",");

        int iNN = Integer.parseInt(metadata[0]);
        int hNN = Integer.parseInt(metadata[1]);
        int oNN = Integer.parseInt(metadata[2]);

        if (iNN == inputNodesNo && hNN == hiddenNeuronNo && oNN == outputNeuronNo) {

            // READ WEIGHTS1
            String weights1String = bufferedReader.readLine();
            String[] weights1Array = weights1String.split(",");
            for (int inputNode = 0; inputNode < inputNodesNo; inputNode++) {
                for (int hiddenNeuron = 0; hiddenNeuron < hiddenNeuronNo; hiddenNeuron++) {
                    try {
                        weights1.set(inputNode, hiddenNeuron, Double.parseDouble(weights1Array[(inputNode * hiddenNeuronNo) + hiddenNeuron]));
                    } catch (ArrayIndexOutOfBoundsException exception) {
                        throw new InvalidWeightFormatException("The network of weights in the file is invalid.");
                    }
                }
            }

            // READ WEIGHTS2
            String weights2String = bufferedReader.readLine();
            String[] weights2Array = weights2String.split(",");
            for (int hiddenNeuron = 0; hiddenNeuron < hiddenNeuronNo + 1; hiddenNeuron++) {
                for (int outputNeuron = 0; outputNeuron < outputNeuronNo; outputNeuron++) {
                    try {
                        weights2.set(hiddenNeuron, outputNeuron, Double.parseDouble(weights2Array[(hiddenNeuron * outputNeuronNo) + outputNeuron]));
                    } catch (ArrayIndexOutOfBoundsException exception) {
                        throw new InvalidWeightFormatException("The network of weights in the file is invalid.");
                    }
                }
            }

            bufferedReader.close();
            JOptionPane.showMessageDialog(null, "Loaded weights from " + source);

        } else throw new InvalidWeightFormatException("The network of weights in the file is invalid.");
    }

    // SAVE WEIGHTS TO GIVEN DESTINATION FILE
    @Override
    public void saveWeights (String destination) throws FileNotFoundException, IOException {
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(destination, false));

        // WRITE METADATA
        bufferedWriter.write(inputNodesNo + "," + hiddenNeuronNo + "," + outputNeuronNo + "\n");

        // WRITE WEIGHTS1
        for (int inputNode = 0; inputNode < inputNodesNo; inputNode++) {
            for (int hiddenNeuron = 0; hiddenNeuron < hiddenNeuronNo; hiddenNeuron++) {
                bufferedWriter.write(Double.toString(weights1.get(inputNode, hiddenNeuron)) + ",");
            }
        }
        bufferedWriter.write("\n");

        // WRITE WEIGHTS2
        for (int hiddenNeuron = 0; hiddenNeuron < hiddenNeuronNo + 1; hiddenNeuron++) {
            for (int outputNeuron = 0; outputNeuron < outputNeuronNo; outputNeuron++) {
                bufferedWriter.write(Double.toString(weights2.get(hiddenNeuron, outputNeuron)) + ",");
            }
        }
        bufferedWriter.close();
        JOptionPane.showMessageDialog(null, "Saved weights to " + destination);
    }
}
