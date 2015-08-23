package com.devankuleindiren.mnist;

import java.awt.*;

public class Image implements Cloneable {

    private int[][] pixels;
    private int width = 0;
    private int height = 0;
    private String label;

    public Image (int width, int height) {
        pixels = new int[height][width];
        this.width = width;
        this.height = height;
    }

    public Image (int[][] pixels, String label) {
        this.pixels = pixels;
        this.height = pixels.length;
        this.width = pixels[0].length;
        this.label = label;
    }

    public void setPixels (int[][] pixels) {
        this.pixels = pixels;
    }

    public void setPixel (int row, int col, int value) {
        if (row >= 0 && row < height
                && col >= 0 && col < pixels[0].length
                && value >= 0) {
            if (value <= 255) pixels[row][col] = value;
            if (value > 255) pixels[row][col] = 255;
        }
    }

    public int getPixel (int row, int col) {
        if (row >= 0 && row < height
                && col >= 0 && col < width) return pixels[row][col];
        return 0;
    }

    public void setLabel (String label) {
        this.label = label;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public String getLabel() {
        return label;
    }

    private int getCell (int row, int col) {
        return pixels[row][col];
    }

    public void draw (Graphics g, int width, int height) {
        int imageW = getWidth();
        int imageH = getHeight();

        double colScale = (double)width/(double)imageW;
        double rowScale = (double)height/(double)imageH;

        for(int col=0; col < imageW; ++col) {
            for(int row=0; row < imageH; ++row) {
                int colPos = (int)(col*colScale);
                int rowPos = (int)(row*rowScale);
                int nextCol = (int)((col+1)*colScale);
                int nextRow = (int)((row+1)*rowScale);

                if (g.hitClip(colPos,rowPos,nextCol-colPos,nextRow-rowPos)) {
                    int color = getCell(row, col);
                    g.setColor(new Color(255 - color, 255 - color, 255 - color));
                    g.fillRect(colPos,rowPos,nextCol-colPos,nextRow-rowPos);
                }
            }
        }
    }

    public double[][] pixelsToVector () {

        double[][] vector = new double[1][(height * width) + 1];

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                vector[0][(row * width) + col] = pixels[row][col];
            }
        }

        vector[0][vector.length - 1] = -1;

        return vector;
    }

    @Override
    public Object clone() {
        try {
            Image clone = (Image) super.clone();
            clone.pixels = new int[height][width];
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    clone.pixels[row][col] = pixels[row][col];
                }
            }
            return clone;

        } catch (CloneNotSupportedException e) {
            System.out.println(e.getMessage());
            return null;
        }
    }

    private void shiftRow (int row, int shift) {
        if (shift > 0) {
            for (int col = width - 1; col >= shift; col--) setPixel(row, col, getPixel(row, col - shift));
            for (int col = 0; col < shift; col++) setPixel(row, col, 0);
        } else {
            for (int col = 0; col <= width - 1 + shift; col++) setPixel(row, col, getPixel(row, col - shift));
            for (int col = width - 1; col > width - 1 + shift; col--) setPixel(row, col, 0);
        }
    }

    private void shiftCol (int col, int shift) {
        if (shift > 0) {
            for (int row = height - 1; row >= shift; row--) setPixel(row, col, getPixel(row - shift, col));
            for (int row = 0; row < shift; row++) setPixel(row, col, 0);
        } else if (shift < 0) {
            for (int row = 0; row <= height - 1 + shift; row++) setPixel(row, col, getPixel(row - shift, col));
            for (int row = height - 1; row > height - 1 + shift; row--) setPixel(row, col, 0);
        }
    }

    public void shiftHorizontal (int shift) {
        for (int row = 0; row < height; row++) shiftRow(row, shift);
    }

    public void shiftVertical (int shift) {
        for (int col = 0; col < width; col++) shiftCol(col, shift);
    }

    public void blur (double blur, int blurRadius) {

        if (blurRadius != 0) {
            int[][] temp = new int[height][width];

            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {

                    if (pixels[row][col] > 80) {
                        temp[row][col] = pixels[row][col];
                    } else {
                        double sum = 0;
                        for (int i = row - blurRadius; i <= row + blurRadius; i++) {
                            for (int j = col - blurRadius; j <= col + blurRadius; j++) {
                                sum += getPixel(i, j);
                            }
                        }
                        temp[row][col] = (int) (blur * sum);
                        if (temp[row][col] > 255) temp[row][col] = 255;
                    }
                }
            }
            pixels = temp;
        }
    }

    public void zoomIn () {

    }

    public void curveHorizontal (double curve, int pinShift) {
        // shiftV (x) = C0.01(x-7)(x-21)
        for (int col = 0; col < width; col++) {
            shiftCol(col, (int) (curve * 0.01 * (col - 10 + pinShift) * (col - 19 + pinShift)));
        }
    }

    public void curveVertical (double curve, int pinShift) {
        // shiftH (y) = C0.01x(x-7)(x-21)
        for (int row = 0; row < height; row++) {
            shiftRow(row, (int) (curve * 0.01 * (row - 10 + pinShift) * (row - 19 + pinShift)));
        }
    }

    private int numberSurroundingPixel (int row, int col, int threshold) {
        int count = 0;
        for (int i = row - 1; i <= row + 1; i++) {
            for (int j = col - 1; j <= col + 1; j++) {
                if (getPixel(i, j) > threshold && i != row && j != col) {
                    count++;
                }
            }
        }
        return count;
    }

    private int sumSurroundingPixel (int row, int col) {
        int sum = 0;
        for (int i = row - 1; i <= row + 1; i++) {
            for (int j = col - 1; j <= col + 1; j++) {
                if (i != row && j != col) {
                    sum += getPixel(i, j);
                }
            }
        }
        return sum;
    }

    private void fillGapsHorizontal () {
        Image copy = (Image) this.clone();
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                if (getPixel(row, col) == 0) {
                    int max1 = 0;
                    int max2 = 0;
                    for (int i = -2; i <= 2; i++) {
                        if (getPixel(row, col + i) > max1) {
                            max2 = max1;
                            max1 = getPixel(row, col + i);
                        } else if (getPixel(row, col + i) > max2) {
                            max2 = getPixel(row, col + i);
                        }
                    }
                    copy.setPixel(row, col, (max1 + max2) / 2);
                }
            }
        }
        this.pixels = copy.pixels;
    }

    private void fillGapsVertical () {
        Image copy = (Image) this.clone();
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                if (getPixel(row, col) == 0) {
                    int max1 = 0;
                    int max2 = 0;
                    for (int i = -2; i <= 2; i++) {
                        if (getPixel(row + i, col) > max1) {
                            max2 = max1;
                            max1 = getPixel(row + i, col);
                        } else if (getPixel(row + i, col) > max2) {
                            max2 = getPixel(row + i, col);
                        }
                    }
                    copy.setPixel(row, col, (max1 + max2) / 2);
                }
            }
        }
        this.pixels = copy.pixels;
    }

    private void copyRow (int fromRow, int toRow, Image from, Image to) {
        for (int col = 0; col < width; col++) {
            to.setPixel(toRow, col, from.getPixel(fromRow, col));
        }
    }

    private void copyCol (int fromCol, int toCol, Image from, Image to) {
        for (int row = 0; row < height; row++) {
            to.setPixel(row, toCol, from.getPixel(row, fromCol));
        }
    }

    public void stretchHorizontal (int center, double stretch) {
        Image copy = (Image) this.clone();
        this.pixels = new int[height][width];

        // col(x) = stretch * (x - center)
        for (int col = 0; col < width; col++) {
            int newCol = col + ((int) (stretch * (col - center)));
            copyCol(col, newCol, copy, this);
        }
        fillGapsHorizontal();
    }

    public void stretchVertical (int center, double stretch) {
        Image copy = (Image) this.clone();
        this.pixels = new int[height][width];

        // row(y) = stretch * (y - center)
        for (int row = 0; row < height; row++) {
            int newRow = row + ((int) (stretch * (row - center)));
            copyRow(row, newRow, copy, this);
        }
        fillGapsVertical();
    }
}
