package com.devankuleindiren.mnist;

import javax.swing.*;

/**
 * Created by Devan Kuleindiren on 22/09/15.
 */
public class FNN2Layer_Train extends SwingWorker <Double, Void> {

    private FNN2Layer fnn2Layer;
    private Matrix weights1;
    private Matrix weights2;

    private int hiddenNeuronNo;
    private int outputNeuronNo;

    private Matrix inputVectors;
    private Matrix targets;
    private double lR = 0.0001;
    private int iterationNo = 1000;
    private Double error = 0.0;

    public FNN2Layer_Train (FNN2Layer fnn2Layer, Matrix weights1, Matrix weights2, int hiddenNeuronNo, int outputNeuronNo, Matrix inputVectors, Matrix targets) {
        this.fnn2Layer = fnn2Layer;
        this.weights1 = weights1;
        this.weights2 = weights2;

        this.hiddenNeuronNo = hiddenNeuronNo;
        this.outputNeuronNo = outputNeuronNo;

        this.inputVectors = inputVectors;
        this.targets = targets;
    }

    // TRAIN THE NEURAL NETWORK IN A BACKGROUND THREAD, SO THE PROGRESS BAR CAN BE UPDATED
    @Override
    protected Double doInBackground () {

        Matrix hiddenActs;
        Matrix outputActs;
        error = 0.0;

        // ERROR TERMS FOR CHANGE IN OUTPUT WEIGHTS
        Matrix deltaO = new Matrix (inputVectors.getHeight(), outputNeuronNo);
        // ERROR TERMS FOR CHANGE IN HIDDEN WEIGHTS
        Matrix deltaH = new Matrix (inputVectors.getHeight(), hiddenNeuronNo);

        try {
            System.out.println("***");
            for (int i = 0; i < iterationNo; i++) {

                // FEED FORWARD
                hiddenActs = inputVectors.feedForwardAndAddBias(weights1);
                outputActs = hiddenActs.multiply(weights2);
                outputActs.applyLogisticActivation();

                error = 0.0;

                for (int j = 0; j < inputVectors.getHeight(); j++) {
                    for (int k = 0; k < outputNeuronNo; k++) {
                        // COMPUTE ERROR
                        error += Math.pow((outputActs.get(j, k) - targets.get(j, k)), 2);

                        // COMPUTE ERROR IN THE OUTPUT NEURONS (LOGISTIC)
                        deltaO.set(j, k, (targets.get(j, k) - outputActs.get(j, k)) * outputActs.get(j, k) * (1 - outputActs.get(j, k)));
                    }
                }

                if (i % 10 == 0) {
                    System.out.println("% Error: " + (error / (inputVectors.getHeight() * outputNeuronNo)) * 100);
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

                setProgress(100 * i / iterationNo);
            }
        } catch (MatrixDimensionMismatchException e) {
            JOptionPane.showMessageDialog(null, e.getMessage());
        }

        return error;
    }

    @Override
    protected void done () {
        fnn2Layer.setWeights(weights1, weights2);
        ControlPanel controlPanel = ControlPanel.getInstance();
        controlPanel.updateTrainingProgressBar(100, Strings.CONTROLPANEL_NEURALNETWORK_TRAININGCOMPLETE);
    }
}
