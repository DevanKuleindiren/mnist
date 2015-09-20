package com.devankuleindiren.mnist;

import javax.swing.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.*;

/**
 * Created by Devan Kuleindiren on 29/08/15.
 */

public class FNN2Layer extends SwingWorker <Double, Void> {

    private static FNN2Layer instance = null;

    private FNN2Layer () {}

    public static FNN2Layer getInstance (int iNN, int hNN, int oNN) {
        if (instance == null) {
            instance = new FNN2Layer();
            inputNodesNo = iNN;
            hiddenNeuronNo = hNN;
            outputNeuronNo = oNN;

            weights1 = new Matrix(inputNodesNo, hiddenNeuronNo);
            weights2 = new Matrix(hiddenNeuronNo + 1, outputNeuronNo);

            initWeights();
        }

        return instance;
    }

    //SPECIFY NUMBER OF NODES IN INPUT, HIDDEN AND OUTPUT LAYERS
    private static int inputNodesNo;
    private static int hiddenNeuronNo;
    private static int outputNeuronNo;

    //GET METHODS FOR THESE NODES
    public int getInputNodesNo () { return inputNodesNo; }
    public int getHiddenNeuronNo () { return hiddenNeuronNo; }
    public int getOutputNeuronNo () { return outputNeuronNo; }

    //2 DIMENSIONAL ARRAY FOR REPRESENTING THE WEIGHTS
    private static Matrix weights1;
    private static Matrix weights2;

    //GET & SET METHODS FOR THESE WEIGHTS
    public double getWeight1 (int i, int j) { return weights1.get(i, j); }
    public double getWeight2 (int i, int j) { return weights2.get(i, j); }
    public void setWeight1 (int i, int j, double newV) { weights1.set(i, j, newV); }
    public void setWeight2 (int i, int j, double newV) { weights2.set(i, j, newV); }

    //INITIALISE EACH WEIGHT TO A RANDOM VALUE, X, IN THE RANGE -N^(-0.5) < X < N^(-0.5)
    // WHERE N IS THE NUMBER OF NODES IN THE LAYER BEFORE THE WEIGHTS
    private static void fillRandom (Matrix matrix, int noOfInputs) {
        for (int row = 0; row < matrix.getHeight(); row++) {
            for (int col = 0; col < matrix.getWidth(); col++) {
                matrix.set(row, col, (Math.random() * (2*(1/Math.pow(noOfInputs, 0.5)))) - (1/Math.pow(noOfInputs, 0.5)));
            }
        }
    }

    //INITIALISE THE WEIGHT ARRAYS
    public static void initWeights() {
        fillRandom(weights1, inputNodesNo);
        fillRandom(weights2, hiddenNeuronNo);
    }

    //SET THE GREATEST ACTIVATION TO 1 AND ALL OTHERS TO 0
    public Matrix rectifyActivations(Matrix activations) {
        for (int row = 0; row < activations.getHeight(); row++) {

            double tempMaxAct = activations.get(row, 0);
            int tempMaxPos = 0;

            for (int col = 0; col < activations.getWidth(); col++) {
                if (activations.get(row, col) > tempMaxAct) {
                    tempMaxAct = activations.get(row, col);
                    tempMaxPos = col;
                }
                activations.set(row, col, 0);
            }
            activations.set(row, tempMaxPos, 1);
        }
        return activations;
    }

    @Override
    public Double doInBackground () {

        Matrix hiddenActs;
        Matrix outputActs;
        error = 0.0;

        //ERROR TERMS FOR CHANGE IN OUTPUT WEIGHTS
        Matrix deltaO = new Matrix (inputVectors.getHeight(), outputNeuronNo);
        //ERROR TERMS FOR CHANGE IN HIDDEN WEIGHTS
        Matrix deltaH = new Matrix (inputVectors.getHeight(), hiddenNeuronNo);

        try {
            System.out.println("***");
            for (int i = 0; i < iterationNo; i++) {

                //FEED FORWARD
                hiddenActs = useNetP1(inputVectors, beta);
                outputActs = useNetP2(hiddenActs, beta);

                error = 0.0;

                for (int j = 0; j < inputVectors.getHeight(); j++) {
                    for (int k = 0; k < outputNeuronNo; k++) {
                        //COMPUTE ERROR
                        error += Math.pow((outputActs.get(j, k) - targets.get(j, k)), 2);

                        //COMPUTE ERROR IN THE OUTPUT NEURONS (LOGISTIC)
                        deltaO.set(j, k, (targets.get(j, k) - outputActs.get(j, k)) * outputActs.get(j, k) * (1 - outputActs.get(j, k)));
                    }
                }

                if ((i-1) % 100 == 0) {
                    System.out.println("% Error: " + (error / (inputVectors.getHeight() * outputNeuronNo)) * 100);
                }

                //COMPUTE ERROR IN THE HIDDEN NEURONS
                for (int j = 0; j < inputVectors.getHeight(); j++) {
                    for (int k = 0; k < hiddenNeuronNo; k++) {

                        double tempWeightErrorSum = 0;
                        for (int l = 0; l < outputNeuronNo; l++) {
                            tempWeightErrorSum += weights2.get(k, l) * deltaO.get(j, l);
                        }

                        deltaH.set(j, k, hiddenActs.get(j, k) * (1 - hiddenActs.get(j, k)) * tempWeightErrorSum);
                    }
                }

                //UPDATE WEIGHTS2
                weights2 = weights2.add((hiddenActs.transpose().multiply(deltaO)).scalarMultiply(lR));

                //UPDATE WEIGHTS1
                weights1 = weights1.add((inputVectors.transpose().multiply(deltaH)).scalarMultiply(lR));

                setProgress(100 * i / iterationNo);
            }
        } catch (MatrixDimensionMismatchException e) {
            JOptionPane.showMessageDialog(null, e.getMessage());
        }

        return error;
    }

    @Override
    public void done () {
        ControlPanel controlPanel = ControlPanel.getInstance();
        controlPanel.updateTrainingProgressBar(100, Strings.CONTROLPANEL_NEURALNETWORK_TRAININGCOMPLETE);
    }

    private Matrix inputVectors;
    private Matrix targets;
    private double lR;
    private double beta;
    private int iterationNo;
    private Double error;

    //TRAIN THE NET USING GIVEN INPUTS & TARGETS, WITH A GIVEN LEARNING RATE, BETA VALUE AND NUMBER OF ITERATIONS
    public double trainNet(Matrix inputVectors, Matrix targets, double lR, double beta, int iterationNo) throws MatrixDimensionMismatchException {
        this.inputVectors = inputVectors;
        this.targets = targets;
        this.lR = lR;
        this.beta = beta;
        this.iterationNo = iterationNo;
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

        return error;
    }

    //PASS AN ARRAY OF INPUT VECTORS THROUGH THE NEURAL NET
    public Matrix useNet(Matrix inputVectors, double beta) throws MatrixDimensionMismatchException {
        Matrix hiddenActs;
        Matrix outputActs;

        //INPUT NODES TO HIDDEN LAYER
        hiddenActs = useNetP1(inputVectors, beta);

        //HIDDEN LAYER TO OUTPUT LAYER
        outputActs = useNetP2(hiddenActs, beta);

        return outputActs;
    }

    //COMPUTE THE ACTIVATIONS OF THE HIDDEN LAYER NODES GIVEN THE INPUT VECTORS
    private Matrix useNetP1(Matrix inputVectors, double beta) throws MatrixDimensionMismatchException {

        Matrix hiddenActsInitial;
        Matrix hiddenActs = new Matrix (inputVectors.getHeight(), hiddenNeuronNo + 1);

        //FIRST PASS
        hiddenActsInitial = inputVectors.multiply(weights1);

        //ACTIVATION FUNCTION
        for (int row = 0; row < inputVectors.getHeight(); row++) {
            for (int col = 0; col < hiddenNeuronNo; col++) {
                hiddenActs.set(row, col, 1/(1+Math.exp(-beta * hiddenActsInitial.get(row, col))));
            }
        }
        //ADD BIAS COLUMN
        for (int row = 0; row < inputVectors.getHeight(); row++) {
            hiddenActs.set(row, hiddenNeuronNo, -1);
        }

        return hiddenActs;
    }

    //COMPUTE THE ACTIVATIONS OF THE OUTPUT LAYER NODES GIVEN THE HIDDEN LAYER ACTIVATIONS
    private Matrix useNetP2(Matrix hiddenActs, double beta) throws MatrixDimensionMismatchException {
        Matrix outputActs;

        //SECOND PASS
        outputActs = hiddenActs.multiply(weights2);

        //ACTIVATION FUNCTION
        for (int row = 0; row < hiddenActs.getHeight(); row++) {
            for (int col = 0; col < outputNeuronNo; col++) {
                outputActs.set(row, col, 1/(1+Math.exp(-beta * outputActs.get(row, col))));
            }
        }

        return outputActs;
    }

    // LOAD WEIGHTS FROM GIVEN SOURCE FILE
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
                        System.out.println(inputNode + ", " + hiddenNeuron);
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
                        System.out.println(hiddenNeuron + ", " + outputNeuron);
                    }
                }
            }

            bufferedReader.close();
            JOptionPane.showMessageDialog(null, "Loaded weights from " + source);

        } else throw new InvalidWeightFormatException("The network of weights in the file is invalid.");
    }

    // SAVE WEIGHTS TO GIVEN DESTINATION FILE
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
