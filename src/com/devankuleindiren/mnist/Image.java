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

    public void shiftHorizontal (int shift) {
        if (shift > 0) {

            for (int col = width - 1; col >= shift; col--) {
                for (int row = 0; row < height; row++) {
                    setPixel(row, col, getPixel(row, col - shift));
                }
            }

            for (int col = 0; col < shift; col++) {
                for (int row = 0; row < height; row++) {
                    setPixel(row, col, 0);
                }
            }

        } else if (shift < 0) {

            for (int col = 0; col <= width - 1 + shift; col++) {
                for (int row = 0; row < height; row++) {
                    setPixel(row, col, getPixel(row, col - shift));
                }
            }

            for (int col = width - 1; col > width - 1 + shift; col--) {
                for (int row = 0; row < height; row++) {
                    setPixel(row, col, 0);
                }
            }
        }
    }

    public void shiftVertical (int shift) {
        if (shift > 0) {

            for (int row = height - 1; row >= shift; row--) {
                for (int col = 0; col < width; col++) {
                    setPixel(row, col, getPixel(row - shift, col));
                }
            }

            for (int row = 0; row < shift; row++) {
                for (int col = 0; col < width; col++) {
                    setPixel(row, col, 0);
                }
            }

        } else if (shift < 0) {

            for (int row = 0; row <= height - 1 + shift; row++) {
                for (int col = 0; col < width; col++) {
                    setPixel(row, col, getPixel(row - shift, col));
                }
            }

            for (int row = height - 1; row > height - 1 + shift; row--) {
                for (int col = 0; col < width; col++) {
                    setPixel(row, col, 0);
                }
            }
        }
    }

    public void blur (double blur, int blurRadius) {

        if (blurRadius != 0) {
            int[][] temp = new int[height][width];

            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
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

            pixels = temp;
        }
    }

}
