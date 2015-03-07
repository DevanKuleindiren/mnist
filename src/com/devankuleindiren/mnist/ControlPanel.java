package com.devankuleindiren.mnist;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;
import java.util.ArrayList;

public class ControlPanel extends JPanel {

    private static ControlPanel instance = null;

    private JButton nextImage;
    private JButton train;
    private JButton classify;
    private JButton save;
    private JButton load;
    private JProgressBar progressBar;
    private JLabel label;
    private JLabel result;

    private int trainIter = 1000;
    private int batchSize = 1000;
    private double lR = 0.001;
    private double beta = 1.0;

    private int inputNodesNo = 785;
    private int hiddenNeuronNo = 15;
    private int outputNeuronNo = 10;

    public static ControlPanel getInstance () {
        if (instance == null) instance = new ControlPanel();

        return instance;
    }

    private ControlPanel() {
        super();

        nextImage = new JButton("Next image");
        train = new JButton("Train");
        classify = new JButton("Classify");
        save = new JButton("Save weights");
        load = new JButton("Load weights");
        progressBar = new JProgressBar(0, trainIter);
        progressBar.setValue(0);
        progressBar.setStringPainted(true);
        label = new JLabel("-");
        result = new JLabel("-");

        GroupLayout layout = new GroupLayout(this);
        this.setLayout(layout);

        layout.setHorizontalGroup(
                layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                                .addComponent(nextImage)
                                .addComponent(label)
                                .addComponent(train)
                                .addComponent(progressBar)
                                .addComponent(classify)
                                .addComponent(result)
                                .addComponent(save)
                                .addComponent(load))
        );
        layout.setVerticalGroup(
                layout.createSequentialGroup()
                        .addComponent(nextImage)
                        .addComponent(label)
                        .addComponent(train)
                        .addComponent(progressBar)
                        .addComponent(classify)
                        .addComponent(result)
                        .addComponent(save)
                        .addComponent(load)
        );

        nextImage.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                DataLoader dataLoader = DataLoader.getInstance();
                Image image = null;
                try {
                    image = dataLoader.next();
                } catch (IOException exception) {
                    System.out.println(exception.getMessage());
                }
                GamePanel gamePanel = GamePanel.getInstance();
                gamePanel.display(image);
                updateLabel(image.getLabel());
            }
        });

        train.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // LOAD DATA FROM DATA LOADER
                System.out.println("LOADING INPUT BATCH...");
                final Batch batch = DataLoader.getInputBatch(batchSize);

                System.out.println("INSTANTIATING DEEPNET...");
                final DeepNet deepNet = DeepNet.getInstance(inputNodesNo, hiddenNeuronNo, outputNeuronNo);

                System.out.println("TRAINING DEEPNET...");
                double error = deepNet.trainNet(batch.getInputs(), batch.getTargets(), lR, beta, trainIter);
                System.out.println();
                System.out.println();
                System.out.println("DEEPNET TRAINED. ERROR: " + Double.toString((error / (batchSize * 10)) * 100) + " %");
            }
        });

        classify.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                GamePanel gamePanel = GamePanel.getInstance();
                Image currentImage = gamePanel.getImage();
                double[][] input = currentImage.pixelsToVector();
                double[][] output;

                DeepNet deepNet = DeepNet.getInstance(inputNodesNo, hiddenNeuronNo, outputNeuronNo);

                output = deepNet.useNet(input, 1.0);
                output = deepNet.rectifyActivations(output);

                for (int i = 0; i < output[0].length; i++) {
                    if (output[0][i] == 1) result.setText("Result: " + i);
                }
            }
        });

        save.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("weights.txt", false));

                    // WRITE METADATA
                    bufferedWriter.write(inputNodesNo + "," + hiddenNeuronNo + "," + outputNeuronNo + "\n");

                    DeepNet deepNet = DeepNet.getInstance(inputNodesNo, hiddenNeuronNo, outputNeuronNo);

                    // WRITE WEIGHTS1
                    for (int inputNode = 0; inputNode < inputNodesNo; inputNode++) {
                        for (int hiddenNeuron = 0; hiddenNeuron < hiddenNeuronNo; hiddenNeuron++) {
                            bufferedWriter.write(Double.toString(deepNet.getWeight1(inputNode, hiddenNeuron)) + ",");
                        }
                    }
                    bufferedWriter.write("\n");

                    // WRITE WEIGHTS2
                    for (int hiddenNeuron = 0; hiddenNeuron < hiddenNeuronNo + 1; hiddenNeuron++) {
                        for (int outputNeuron = 0; outputNeuron < outputNeuronNo; outputNeuron++) {
                            bufferedWriter.write(Double.toString(deepNet.getWeight2(hiddenNeuron, outputNeuron)) + ",");
                        }
                    }

                    bufferedWriter.close();
                } catch (IOException exception) {
                    System.out.println("Could not save weights.");
                }
            }
        });

        load.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    BufferedReader bufferedReader = new BufferedReader(new FileReader("weights.txt"));

                    String metadataLine = bufferedReader.readLine();
                    String[] metadata = metadataLine.split(",");

                    int iNN = Integer.parseInt(metadata[0]);
                    int hNN = Integer.parseInt(metadata[1]);
                    int oNN = Integer.parseInt(metadata[2]);

                    if (iNN == inputNodesNo && hNN == hiddenNeuronNo && oNN == outputNeuronNo) {

                        DeepNet deepNet = DeepNet.getInstance(inputNodesNo, hiddenNeuronNo, outputNeuronNo);

                        // READ WEIGHTS1
                        String weights1String = bufferedReader.readLine();
                        String[] weights1 = weights1String.split(",");
                        for (int inputNode = 0; inputNode < inputNodesNo; inputNode++) {
                            for (int hiddenNeuron = 0; hiddenNeuron < hiddenNeuronNo; hiddenNeuron++) {
                                try {
                                    deepNet.setWeight1(inputNode, hiddenNeuron, Double.parseDouble(weights1[(inputNode * hiddenNeuronNo) + hiddenNeuron]));
                                } catch (ArrayIndexOutOfBoundsException exception) {
                                    System.out.println(inputNode + ", " + hiddenNeuron);
                                }
                            }
                        }

                        // READ WEIGHTS2
                        String weights2String = bufferedReader.readLine();
                        String[] weights2 = weights2String.split(",");
                        for (int hiddenNeuron = 0; hiddenNeuron < hiddenNeuronNo + 1; hiddenNeuron++) {
                            for (int outputNeuron = 0; outputNeuron < outputNeuronNo; outputNeuron++) {
                                try {
                                    deepNet.setWeight2(hiddenNeuron, outputNeuron, Double.parseDouble(weights2[(hiddenNeuron * outputNeuronNo) + outputNeuron]));
                                } catch (ArrayIndexOutOfBoundsException exception) {
                                    System.out.println(hiddenNeuron + ", " + outputNeuron);
                                }
                            }
                        }

                        bufferedReader.close();

                    } else throw new InvalidWeightFormatException("The network of weights in the file is invalid.");

                } catch (IOException exception) {
                    System.out.println("Could not load weights.");
                } catch (InvalidWeightFormatException exception) {
                    System.out.println(exception.getMessage());
                }
            }
        });
    }

    public void updateLabel (String label) {
        if (this.label != null) {
            this.label.setText("Digit: " + label);
        }
    }

    public void updateProgress (int progress) {
        progressBar.setValue(progress);
        progressBar.repaint();
    }
}


