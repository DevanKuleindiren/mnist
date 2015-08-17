package com.devankuleindiren.mnist;

/**
 * Created by Devan Kuleindiren on 18/06/2015.
 */
public class Diagnostics {

    public static void printMatrix (double[][] matrix) {
        for (int row = 0; row < matrix.length; row++) {
            for (int col = 0; col < matrix[0].length; col++) {
                System.out.println("Matrix[" + row + "][" + col + "] = " + matrix[row][col]);
            }
        }
    }

}
