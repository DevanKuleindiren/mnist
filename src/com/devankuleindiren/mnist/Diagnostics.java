package com.devankuleindiren.mnist;

/**
 * Created by Devan Kuleindiren on 18/06/2015.
 */
public class Diagnostics {

    public static void printMatrix (Matrix matrix, String label) {
        System.out.println("MATRIX: " + label);
        for (int row = 0; row < matrix.getHeight(); row++) {
            for (int col = 0; col < matrix.getWidth(); col++) {
                System.out.println("Matrix[" + row + "][" + col + "] = " + matrix.get(row, col));
            }
        }
    }

    public static void printMatrixPretty (Matrix matrix, String label, int limit) {
        System.out.print("\n");
        System.out.println("MATRIX: " + label);
        for (int row = 0; row < matrix.getHeight() && row < limit; row++) {
            for (int col = 0; col < matrix.getWidth(); col++) {
                System.out.printf("%.3f, ", matrix.get(row, col));
            }
            System.out.print("\n");
        }
        System.out.print("\n");
    }

}
