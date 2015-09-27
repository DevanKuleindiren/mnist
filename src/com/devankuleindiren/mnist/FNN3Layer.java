package com.devankuleindiren.mnist;

import javax.swing.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.*;

/**
 * Created by Devan Kuleindiren on 27/09/15.
 */
public class FNN3Layer extends SwingWorker<Double, Void> implements NeuralNetwork {

    public FNN3Layer () {
        initWeights();
    }

    // SPECIFY NUMBER OF NODES IN INPUT, HIDDEN AND OUTPUT LAYERS
    private int inputNodesNo = 785;
    private int hiddenNeuronNo1 = 28;
    private int hiddenNeuronNo2 = 20;
    private int outputNeuronNo = 14;

    // WEIGHT MATRICES
    private Matrix weights_I_H1 = new Matrix(inputNodesNo, hiddenNeuronNo1);
    private Matrix weights_H1_H2 = new Matrix(hiddenNeuronNo1 + 1, hiddenNeuronNo2);
    private Matrix weights_H2_O = new Matrix(hiddenNeuronNo2 + 1, outputNeuronNo);

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
        fillRandom(weights_I_H1, inputNodesNo);
        fillRandom(weights_H1_H2, hiddenNeuronNo1);
        fillRandom(weights_H2_O, hiddenNeuronNo2);
    }

    private Matrix inputVectors;
    private Matrix targets;
    private double lR = 0.0001;
    private int iterationNo = 2000;
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

        try {
            for (int iteration = 0; iteration < iterationNo; iteration++) {

                // FEED FORWARD
                Matrix hiddenActs1;
                Matrix hiddenActs2;
                Matrix outputActs;

                // INPUT NODES TO HIDDEN LAYER 1
                hiddenActs1 = inputVectors.feedForwardAndAddBias(weights_I_H1);

                // HIDDEN LAYER 1 TO HIDDEN LAYER 2
                hiddenActs2 = hiddenActs1.feedForwardAndAddBias(weights_H1_H2);

                // HIDDEN LAYER TO OUTPUT LAYER
                outputActs = hiddenActs2.multiply(weights_H2_O);
                outputActs.applyLogisticActivation();

                error = 0.0;

                for (int j = 0; j < inputVectors.getHeight(); j++) {
                    for (int k = 0; k < outputNeuronNo; k++) {
                        // COMPUTE ERROR
                        error += Math.pow((outputActs.get(j, k) - targets.get(j, k)), 2);
                    }
                }
                if (iteration % 100 == 0) {
                    System.out.println("Error measure: " + (error / (inputVectors.getHeight() * outputNeuronNo)) * 100);
                }

                // PREPARE ONES MATRICES FOR CALCULATION
                Matrix OOnes = Matrix.onesMatrix(inputVectors.getHeight(), outputNeuronNo);
                Matrix H2Ones = Matrix.onesMatrix(inputVectors.getHeight(), hiddenNeuronNo2);
                Matrix H1Ones  = Matrix.onesMatrix(inputVectors.getHeight(), hiddenNeuronNo1);

                // REMOVES BIAS COLUMNS FOR CALCULATIONS
                Matrix hiddenActs1NoBias = hiddenActs1.removeBiasCol();
                Matrix hiddenActs2NoBias = hiddenActs2.removeBiasCol();

                // CALCULATE ALL DELTA TERMS
                Matrix deltaO = outputActs.subtract(targets);
                Matrix deltaH2 = deltaO.multiply(weights_H2_O.removeBiasRow().transpose()).multiplyEach(hiddenActs2NoBias.multiplyEach(H2Ones.subtract(hiddenActs2NoBias)));
                Matrix deltaH1  = deltaH2.multiply(weights_H1_H2.removeBiasRow().transpose()).multiplyEach(hiddenActs1NoBias.multiplyEach(H1Ones.subtract(hiddenActs1NoBias)));

                // UPDATE WEIGHTS H2 TO O
                weights_H2_O = weights_H2_O.subtract((hiddenActs2.transpose().multiply(deltaO)).scalarMultiply(lR));

                // UPDATE WEIGHTS H1 TO H2
                weights_H1_H2 = weights_H1_H2.subtract((hiddenActs1.transpose().multiply(deltaH2)).scalarMultiply(lR));

                // UPDATE WEIGHTS I TO H1
                weights_I_H1 = weights_I_H1.subtract((inputVectors.transpose().multiply(deltaH1)).scalarMultiply(lR));

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
        Matrix hiddenActs1;
        Matrix hiddenActs2;
        Matrix outputActs;

        // INPUT NODES TO HIDDEN LAYER 1
        hiddenActs1 = inputVectors.feedForwardAndAddBias(weights_I_H1);

        // HIDDEN LAYER 1 TO HIDDEN LAYER 2
        hiddenActs2 = hiddenActs1.feedForwardAndAddBias(weights_H1_H2);

        // HIDDEN LAYER TO OUTPUT LAYER
        outputActs = hiddenActs2.multiply(weights_H2_O);
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
        int hNN1 = Integer.parseInt(metadata[1]);
        int hNN2 = Integer.parseInt(metadata[2]);
        int oNN = Integer.parseInt(metadata[3]);

        if (iNN == inputNodesNo && hNN1 == hiddenNeuronNo1 && hNN2 == hiddenNeuronNo2 && oNN == outputNeuronNo) {

            // READ WEIGHTS I TO H1
            String weightsIH1String = bufferedReader.readLine();
            String[] weightsIH1Array = weightsIH1String.split(",");
            for (int inputNode = 0; inputNode < inputNodesNo; inputNode++) {
                for (int hiddenNeuron1 = 0; hiddenNeuron1 < hiddenNeuronNo1; hiddenNeuron1++) {
                    try {
                        weights_I_H1.set(inputNode, hiddenNeuron1, Double.parseDouble(weightsIH1Array[(inputNode * hiddenNeuronNo1) + hiddenNeuron1]));
                    } catch (ArrayIndexOutOfBoundsException exception) {
                        throw new InvalidWeightFormatException("The network of weights in the file is invalid.");
                    }
                }
            }

            // READ WEIGHTS H1 TO H2
            String weightsH1H2String = bufferedReader.readLine();
            String[] weightsH1H2Array = weightsH1H2String.split(",");
            for (int hiddenNeuron1 = 0; hiddenNeuron1 < hiddenNeuronNo1 + 1; hiddenNeuron1++) {
                for (int hiddenNeuron2 = 0; hiddenNeuron2 < hiddenNeuronNo2; hiddenNeuron2++) {
                    try {
                        weights_H1_H2.set(hiddenNeuron1, hiddenNeuron2, Double.parseDouble(weightsH1H2Array[(hiddenNeuron1 * hiddenNeuronNo2) + hiddenNeuron2]));
                    } catch (ArrayIndexOutOfBoundsException exception) {
                        throw new InvalidWeightFormatException("The network of weights in the file is invalid.");
                    }
                }
            }

            // READ WEIGHTS2
            String weightsH2OString = bufferedReader.readLine();
            String[] weightsH2OArray = weightsH2OString.split(",");
            for (int hiddenNeuron2 = 0; hiddenNeuron2 < hiddenNeuronNo2 + 1; hiddenNeuron2++) {
                for (int outputNeuron = 0; outputNeuron < outputNeuronNo; outputNeuron++) {
                    try {
                        weights_H2_O.set(hiddenNeuron2, outputNeuron, Double.parseDouble(weightsH2OArray[(hiddenNeuron2 * outputNeuronNo) + outputNeuron]));
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
        bufferedWriter.write(inputNodesNo + "," + hiddenNeuronNo1 + "," + hiddenNeuronNo2 + "," + outputNeuronNo + "\n");

        // WRITE WEIGHTS I TO H1
        for (int inputNode = 0; inputNode < inputNodesNo; inputNode++) {
            for (int hiddenNeuron1 = 0; hiddenNeuron1 < hiddenNeuronNo1; hiddenNeuron1++) {
                bufferedWriter.write(Double.toString(weights_I_H1.get(inputNode, hiddenNeuron1)) + ",");
            }
        }
        bufferedWriter.write("\n");

        // WRITE WEIGHTS H1 TO H2
        for (int hiddenNeuron1 = 0; hiddenNeuron1 < hiddenNeuronNo1 + 1; hiddenNeuron1++) {
            for (int hiddenNeuron2 = 0; hiddenNeuron2 < hiddenNeuronNo2; hiddenNeuron2++) {
                bufferedWriter.write(Double.toString(weights_H1_H2.get(hiddenNeuron1, hiddenNeuron2)) + ",");
            }
        }
        bufferedWriter.write("\n");

        // WRITE WEIGHTS2
        for (int hiddenNeuron2 = 0; hiddenNeuron2 < hiddenNeuronNo2 + 1; hiddenNeuron2++) {
            for (int outputNeuron = 0; outputNeuron < outputNeuronNo; outputNeuron++) {
                bufferedWriter.write(Double.toString(weights_H2_O.get(hiddenNeuron2, outputNeuron)) + ",");
            }
        }
        bufferedWriter.close();
        JOptionPane.showMessageDialog(null, "Saved weights to " + destination);
    }
}

