package com.devankuleindiren.mnist;

import javax.swing.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.*;

/**
 * Created by Devan Kuleindiren on 29/08/15.
 */

public class FNN2Layer extends SwingWorker <Double, Void> implements NeuralNetwork {

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

    private Matrix inputVectors;
    private Matrix targets;
    private double lR = 0.0001;
    private int iterationNo = 1000;
    private Double error;

    // TRAIN THE NET USING GIVEN INPUTS & TARGETS, WITH A GIVEN LEARNING RATE AND NUMBER OF ITERATIONS
    @Override
    public void train (Matrix inputVectors, Matrix targets) throws MatrixDimensionMismatchException {
        this.inputVectors = inputVectors;
        this.targets = targets;
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

    // TRAIN THE NEURAL NETWORK IN A BACKGROUND THREAD, SO THE PROGRESS BAR CAN BE UPDATED
    @Override
    protected Double doInBackground () {

        Matrix hiddenActs;
        Matrix outputActs;

        // ERROR TERMS FOR CHANGE IN OUTPUT WEIGHTS
        Matrix deltaO = new Matrix (inputVectors.getHeight(), outputNeuronNo);
        // ERROR TERMS FOR CHANGE IN HIDDEN WEIGHTS
        Matrix deltaH = new Matrix (inputVectors.getHeight(), hiddenNeuronNo);

        try {
            for (int iteration = 0; iteration < iterationNo; iteration++) {

                // FEED FORWARD
                hiddenActs = feedForwardP1(inputVectors);
                outputActs = feedForwardP2(hiddenActs);

                error = 0.0;

                for (int j = 0; j < inputVectors.getHeight(); j++) {
                    for (int k = 0; k < outputNeuronNo; k++) {
                        // COMPUTE ERROR
                        error += Math.pow((outputActs.get(j, k) - targets.get(j, k)), 2);

                        // COMPUTE ERROR IN THE OUTPUT NEURONS (LOGISTIC)
                        deltaO.set(j, k, (targets.get(j, k) - outputActs.get(j, k)) * outputActs.get(j, k) * (1 - outputActs.get(j, k)));
                    }
                }

                if (iteration % 10 == 0) {
                    System.out.println("Error measure: " + (error / (inputVectors.getHeight() * outputNeuronNo)) * 100);
                }

                // COMPUTE ERROR IN THE HIDDEN NEURONS
                for (int j = 0; j < inputVectors.getHeight(); j++) {
                    for (int k = 0; k < hiddenNeuronNo; k++) {

                        double tempWeightErrorSum = 0;
                        for (int l = 0; l < outputNeuronNo; l++) {
                            tempWeightErrorSum += weights2.get(k, l) * deltaO.get(j, l);
                        }

                        deltaH.set(j, k, hiddenActs.get(j, k) * (1 - hiddenActs.get(j, k)) * tempWeightErrorSum);
                    }
                }

                // UPDATE WEIGHTS2
                weights2 = weights2.add((hiddenActs.transpose().multiply(deltaO)).scalarMultiply(lR));

                // UPDATE WEIGHTS1
                weights1 = weights1.add((inputVectors.transpose().multiply(deltaH)).scalarMultiply(lR));

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
