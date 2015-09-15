package com.devankuleindiren.mnist;

import java.io.*;

public class DataLoader {

    private static BufferedReader loadImageReader;
    private static String loadImageFileName;

    private static DataLoader instance = null;
    private DataLoader () {}

    public static DataLoader getInstance () {
        if (instance == null) {
            instance = new DataLoader();
            try {
                loadImageReader = new BufferedReader(new FileReader("mnist_test.csv"));
                loadImageFileName = "mnist_test.csv";
            } catch (IOException e) {
                System.out.println(e.getMessage());
            }
        }
        return instance;
    }

    public static Batch getInputBatch (int batchSize, String fileName) throws IOException {

        double[][] inputs = new double[batchSize][785];
        double[][] targets = new double[batchSize][14];

        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));

            for (int i = 0; i < batchSize; i++) {
                String temp = reader.readLine();
                String[] values = temp.split(",");

                for (int j = 1; j < values.length; j++) inputs[i][j - 1] = (Double.parseDouble(values[j]) / 255.0);
                inputs[i][784] = -1;

                int target;
                if (values[0].equals("+")) {
                    target = 10;
                } else if (values[0].equals("-")) {
                    target = 11;
                } else if (values[0].equals("x")) {
                    target = 12;
                } else if (values[0].equals("d")) {
                    target = 13;
                } else {
                    target = Integer.parseInt(values[0]);
                }
                targets[i][target] = 1;
            }

            return new Batch(inputs, targets);

        } catch (FileNotFoundException e) {
            throw new IOException(fileName + " cannot be found.");
        } catch (EOFException e) {
            throw new IOException("Reached end of file: " + fileName + ". Requested input batch too large.");
        } catch (IOException e) {
            throw new IOException("Error reading batch from: " + fileName);
        }
    }

    public static MatrixBatch getMatrixInputBatch (int batchSize, String fileName) throws IOException {

        Matrix inputs = new Matrix(batchSize, 785);
        Matrix targets = new Matrix(batchSize, 14);

        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));

            for (int i = 0; i < batchSize; i++) {
                String temp = reader.readLine();
                String[] values = temp.split(",");

                for (int j = 1; j < values.length; j++) inputs.set(i, j - 1, Double.parseDouble(values[j]) / 255.0);
                inputs.set(i, 784, -1);

                int target;
                if (values[0].equals("+")) {
                    target = 10;
                } else if (values[0].equals("-")) {
                    target = 11;
                } else if (values[0].equals("x")) {
                    target = 12;
                } else if (values[0].equals("d")) {
                    target = 13;
                } else {
                    target = Integer.parseInt(values[0]);
                }
                targets.set(i, target, 1);
            }

            return new MatrixBatch(inputs, targets);

        } catch (FileNotFoundException e) {
            throw new IOException(fileName + " cannot be found.");
        } catch (EOFException e) {
            throw new IOException("Reached end of file: " + fileName + ". Requested input batch too large.");
        } catch (IOException e) {
            throw new IOException("Error reading batch from: " + fileName);
        } catch (NullPointerException e) {
            throw new IOException("Reached end of file: " + fileName + ". Requested input batch too large.");
        }
    }

    public static Image next (String fileName) throws IOException {

        if (!fileName.equals(loadImageFileName)) {
            try {
                loadImageReader = new BufferedReader(new FileReader(fileName));
                loadImageFileName = fileName;
            } catch (FileNotFoundException e) {
                throw new IOException(fileName + " cannot be found.");
            }
        }

        try {
            String temp = loadImageReader.readLine();
            String[] values = temp.split(",");

            int[][] data = new int[28][28];

            String label = values[0];

            for (int i = 1; i < values.length; i++) {
                data[(i - 1) / 28][(i - 1) % 28] = Integer.parseInt(values[i]);
            }

            return new Image(data, label);

        } catch (EOFException e) {
            throw new IOException("Reached the end of " + fileName);
        } catch (IOException e) {
            throw new IOException("Error reading from " + fileName);
        }
    }

}
