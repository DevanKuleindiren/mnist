package com.devankuleindiren.mnist;

/**
 * Created by Devan Kuleindiren on 21/09/15.
 */
public class Kernel {

    private int height;
    private int width;
    private double[][] values;
    private double biasWeight;

    public Kernel (int height, int width) {
        this.height = height;
        this.width = width;
        this.values = new double[height][width];
    }

    public int getHeight() { return height; }
    public int getWidth() { return width; }

    public double getBiasWeight () { return biasWeight; }
    public void setBiasWeight (double newWeight) { biasWeight = newWeight; }

    public double get (int row, int col) {
        if (row >= 0 && row < height && col >= 0 && col < width) {
            return values[row][col];
        }
        return 0;
    }

    public void set (int row, int col, double newVal) {
        if (row >= 0 && row < height && col >= 0 && col < width) {
            values[row][col] = newVal;
        }
    }

    public Matrix convolute (Matrix input) throws MatrixDimensionMismatchException {

        if (height <= input.getHeight() && width <= input.getWidth()) {
            Matrix result = new Matrix(input.getHeight() - height + 1, input.getWidth() - width + 1);

            for (int row = 0; row < result.getHeight(); row++) {
                for (int col = 0; col < result.getWidth(); col++) {
                    double sum = 0;

                    for (int kRow = 0; kRow < height; kRow++) {
                        for (int kCol = 0; kCol < width; kCol++) {
                            sum += input.get(row + kRow, col + kCol) * values[kRow][kCol];
                        }
                    }
                    sum += -1 * biasWeight;
                    result.set(row, col, sum);
                }
            }

            return result;
        } else {
            throw new MatrixDimensionMismatchException("image convolution");
        }
    }
}