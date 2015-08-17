package com.devankuleindiren.mnist;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;


/* THIS CLASS PROVIDES METHODS TO TRAIN AND USE THE NEURAL NETWORK */ 
public class DeepNet {

	private static DeepNet instance = null;
	private static ControlPanel controlPanel;

	private DeepNet () {}

	public static DeepNet getInstance (int iNN, int hNN, int oNN) {
		if (instance == null) {
			instance = new DeepNet();
			inputNodesNo = iNN;
			hiddenNeuronNo = hNN;
			outputNeuronNo = oNN;

			weights1 = new double[inputNodesNo][hiddenNeuronNo];
			weights2 = new double[hiddenNeuronNo+1][outputNeuronNo];

			initWeights();

			controlPanel = ControlPanel.getInstance();
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
	private static double weights1[][];
	private static double weights2[][];

	//GET & SET METHODS FOR THESE WEIGHTS
	public double getWeight1 (int i, int j) { return weights1[i][j]; }
	public double getWeight2 (int i, int j) { return weights2[i][j]; }
	public void setWeight1 (int i, int j, double newV) { weights1[i][j] = newV; }
	public void setWeight2 (int i, int j, double newV) { weights2[i][j] = newV; }
	
	//INITIALISE EACH WEIGHT TO A RANDOM VALUE, X, IN THE RANGE -N^(-0.5) < X < N^(-0.5) 
	// WHERE N IS THE NUMBER OF NODES IN THE LAYER BEFORE THE WEIGHTS
	private static void fillRandom(double array[][], int noOfInputs) {
		for (int i = 0; i < array.length; i++) {
			for (int j = 0; j < array[0].length; j++) {
				array[i][j] = (Math.random() * (2*(1/Math.pow(noOfInputs, 0.5)))) - (1/Math.pow(noOfInputs, 0.5));
			}
		}
	}
	
	//INITIALISE THE WEIGHT ARRAYS
	public static void initWeights() {
		fillRandom(weights1, inputNodesNo);
		fillRandom(weights2, hiddenNeuronNo);	
	}
	
	//SET THE GREATEST ACTIVATION TO 1 AND ALL OTHERS TO 0
	public double[][] rectifyActivations(double activations[][]) {
		for (int i = 0; i < activations.length; i++) {
			
			double tempMaxAct = activations[i][0];
			int tempMaxPos = 0;
			
			for (int j = 0; j < activations[0].length; j++) {
				if (activations[i][j] > tempMaxAct) {
					tempMaxAct = activations[i][j];
					tempMaxPos = j;
				}
				activations[i][j] = 0;
			}	
			activations[i][tempMaxPos] = 1;
		}
		return activations;
	}
	
	//TRAIN THE NET USING GIVEN INPUTS & TARGETS, WITH A GIVEN LEARNING RATE, BETA VALUE AND NUMBER OF ITERATIONS
	public double trainNet(double inputVectors[][], double targets[][], double lR, double beta, int iterationNo) {
		
		double hiddenActs[][];
		double outputActs[][];
		double error = 0;
		
		//ERROR TERMS FOR CHANGE IN OUTPUT WEIGHTS
		double deltaO[][] = new double[inputVectors.length][outputNeuronNo];
		//ERROR TERMS FOR CHANGE IN HIDDEN WEIGHTS
		double deltaH[][] = new double[inputVectors.length][hiddenNeuronNo];
		
		System.out.println("***");
		for (int i = 0; i < iterationNo; i++) {
			
			//FEED FORWARD
			hiddenActs = useNetP1(inputVectors, beta);
			outputActs = useNetP2(hiddenActs, beta);
			
			error = 0;
			
			for (int j = 0; j < inputVectors.length; j++) {
				for (int k = 0; k < outputNeuronNo; k++) {
					//COMPUTE ERROR
					error += Math.pow((outputActs[j][k]-targets[j][k]), 2);

					//COMPUTE ERROR IN THE OUTPUT NEURONS (LOGISTIC)
					deltaO[j][k] = (targets[j][k] - outputActs[j][k]) * outputActs[j][k] * (1 - outputActs[j][k]);
				}
			}

			if ((i-1) % 100 == 0) {
				System.out.println("% Error: " + (error / (inputVectors.length * outputNeuronNo)) * 100);
			}
			
			//COMPUTE ERROR IN THE HIDDEN NEURONS
			for (int j = 0; j < inputVectors.length; j++) {
				for (int k = 0; k < hiddenNeuronNo; k++) {
					
					double tempWeightErrorSum = 0;
					for (int l = 0; l < outputNeuronNo; l++) {
						tempWeightErrorSum += weights2[k][l] * deltaO[j][l];
					}
					
					deltaH[j][k] = hiddenActs[j][k] * (1 - hiddenActs[j][k]) * tempWeightErrorSum;
				}
			}
			
			//UPDATE WEIGHTS2
			RealMatrix weights2M = MatrixUtils.createRealMatrix(weights2);
			RealMatrix deltaOM = MatrixUtils.createRealMatrix(deltaO);
			RealMatrix hiddenActsM = MatrixUtils.createRealMatrix(hiddenActs);
			
			weights2M = weights2M.add((hiddenActsM.transpose().multiply(deltaOM)).scalarMultiply(lR));
			weights2 = weights2M.getData();
			
			//UPDATE WEIGHTS1
			RealMatrix weights1M = MatrixUtils.createRealMatrix(weights1);
			RealMatrix deltaHM = MatrixUtils.createRealMatrix(deltaH);
			RealMatrix inputVectorsM = MatrixUtils.createRealMatrix(inputVectors);
			
			weights1M = weights1M.add((inputVectorsM.transpose().multiply(deltaHM)).scalarMultiply(lR));
			weights1 = weights1M.getData();
		}
		
		return error;
	}
	
	//PASS AN ARRAY OF INPUT VECTORS THROUGH THE NEURAL NET
	public double[][] useNet(double inputVectors[][], double beta) {
		double hiddenActs[][];
		double outputActs[][];

		//INPUT NODES TO HIDDEN LAYER
		hiddenActs = useNetP1(inputVectors, beta);

		//HIDDEN LAYER TO OUTPUT LAYER
		outputActs = useNetP2(hiddenActs, beta);
		
		return outputActs;
	}
	
	//COMPUTE THE ACTIVATIONS OF THE HIDDEN LAYER NODES GIVEN THE INPUT VECTORS
	private double[][] useNetP1(double inputVectors[][], double beta) {
		
		double hiddenActsInitial[][];
		double hiddenActs[][] = new double[inputVectors.length][hiddenNeuronNo + 1];
		
		RealMatrix inputVectorsM = MatrixUtils.createRealMatrix(inputVectors);
		RealMatrix weights1M = MatrixUtils.createRealMatrix(weights1);
		RealMatrix hiddenActsM;

		//FIRST PASS
		hiddenActsM = inputVectorsM.multiply(weights1M);
		hiddenActsInitial = hiddenActsM.getData();
		
		//ACTIVATION FUNCTION
		for (int i = 0; i < inputVectors.length; i++) {
			for (int j = 0; j < hiddenNeuronNo; j++) {
				hiddenActs[i][j] = 1/(1+Math.exp(-beta * hiddenActsInitial[i][j]));
			}
		}
		//ADD BIAS COLUMN
		for (int i = 0; i < inputVectors.length; i++) {
			hiddenActs[i][hiddenNeuronNo] = -1;
		}
		
		return hiddenActs;
	}
	
	//COMPUTE THE ACTIVATIONS OF THE OUTPUT LAYER NODES GIVEN THE HIDDEN LAYER ACTIVATIONS
	private double[][] useNetP2(double hiddenActs[][], double beta) {
		double outputActs[][];

		RealMatrix hiddenActsM = MatrixUtils.createRealMatrix(hiddenActs);
		RealMatrix weights2M = MatrixUtils.createRealMatrix(weights2);
		RealMatrix outputActsM;

		//SECOND PASS
		outputActsM = hiddenActsM.multiply(weights2M);
		outputActs = outputActsM.getData();
				
		//ACTIVATION FUNCTION
		for (int i = 0; i < hiddenActs.length; i++) {
			for (int j = 0; j < outputNeuronNo; j++) {
				outputActs[i][j] = 1/(1+Math.exp(-beta * outputActs[i][j]));
			}
		}
		
		return outputActs;
	}
}
