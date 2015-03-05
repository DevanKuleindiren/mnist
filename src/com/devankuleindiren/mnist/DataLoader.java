package com.devankuleindiren.mnist;

import java.io.*;

public class DataLoader {

    private static BufferedReader reader;
    private static BufferedReader trainReader;
    private static DataLoader instance = null;
    private DataLoader () {}

    public static DataLoader getInstance () {
        if (instance == null) {
            instance = new DataLoader();
            try {
                reader = new BufferedReader(new FileReader("mnist_test.csv"));
                trainReader = new BufferedReader(new FileReader("mnist_train.csv"));
            } catch (IOException e) {
                System.out.println(e.getMessage());
            }
        }

        return instance;
    }

    public static Batch getInputBatch (int batchSize) {
        double[][] inputs = new double[batchSize][785];
        double[][] targets = new double[batchSize][10];

        for (int i = 0; i < batchSize; i++) {
            try {
                String temp = trainReader.readLine();
                String[] values = temp.split(",");

                for (int j = 1; j < values.length; j++) inputs[i][j - 1] = (Double.parseDouble(values[j]) / 255.0);
                inputs[i][784] = -1;

                int target = Integer.parseInt(values[0]);
                targets[i][target] = 1;

            } catch (IOException e) {
                break;
                //TODO: HANDLE THIS PROPERLY
            } catch (NumberFormatException e) {
                System.out.println(e.getMessage());
            }
        }

        return new Batch(inputs, targets);
    }

    public static Image next () throws IOException {
        String temp = reader.readLine();
        String[] values = temp.split(",");

        int[][] data = new int[28][28];

        String label = values[0];

        for (int i = 1; i < values.length; i++) {
            data[(i - 1) / 28][(i - 1) % 28] = Integer.parseInt(values[i]);
        }

        return new Image(data, label);
    }

}
