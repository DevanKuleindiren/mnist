package com.devankuleindiren.mnist;

/**
 * Created by Devan Kuleindiren on 21/09/15.
 */
public class Kernel extends Matrix implements Cloneable {

    private double biasWeight;

    public Kernel (int height, int width) {
        super(height, width);
    }

    public double getBiasWeight () { return biasWeight; }
    public void setBiasWeight (double newWeight) { biasWeight = newWeight; }

    public Matrix convolute (Matrix input) throws MatrixDimensionMismatchException {

        if (super.getHeight() <= input.getHeight() && super.getWidth() <= input.getWidth()) {
            Matrix result = new Matrix(input.getHeight() - super.getHeight() + 1, input.getWidth() - super.getWidth() + 1);

            for (int row = 0; row < result.getHeight(); row++) {
                for (int col = 0; col < result.getWidth(); col++) {
                    double sum = 0;

                    for (int kRow = 0; kRow < super.getHeight(); kRow++) {
                        for (int kCol = 0; kCol < super.getWidth(); kCol++) {
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

    @Override
    public Object clone () {
        Kernel clone = (Kernel) super.clone();
        clone.biasWeight = this.biasWeight;

        return clone;
    }
}