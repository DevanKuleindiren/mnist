package com.devankuleindiren.mnist;

import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * Created by Devan Kuleindiren on 21/09/15.
 */
public interface NeuralNetwork {
    void train (Matrix inputVectors, Matrix targets) throws MatrixDimensionMismatchException;
    Matrix feedForward (Matrix inputVectors) throws MatrixDimensionMismatchException;
    void loadWeights (String source) throws FileNotFoundException, IOException, InvalidWeightFormatException;
    void saveWeights (String destination) throws FileNotFoundException, IOException;
}
