package com.devankuleindiren.mnist;

/**
 * Created by Devan Kuleindiren on 21/09/15.
 */
public class Kernel {

    private int height;
    private int width;
    double[][] values;

    public Kernel (int height, int width) {
        this.height = height;
        this.width = width;
        values = new double[height][width];
    }

    public Matrix apply (Matrix input) {
        Matrix result = new Matrix(input.getHeight() - height + 1, input.getWidth() - width + 1);

        // TODO

        return result;
    }
}