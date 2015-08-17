package com.devankuleindiren.mnist;

import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.border.EtchedBorder;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;

public class ControlPanel extends JPanel {

    private static ControlPanel instance = null;

    private JButton nextImage;
    private JButton train;
    private JLabel label;
    private JLabel progress;
    private JButton classify;
    private JLabel result;
    private JButton save;
    private JButton load;
    private JButton erase;

    private int trainIter = 1000;
    private int batchSize = 1000;
    private double lR = 0.001;
    private double beta = 1.0;

    private int inputNodesNo = 785;
    private int hiddenNeuronNo = 15;
    private int outputNeuronNo = 10;

    public static ControlPanel getInstance() {
        if (instance == null) instance = new ControlPanel();

        return instance;
    }

    private ControlPanel() {
        super();

        // SECTIONS OF THE CONTROL PANEL

        JPanel loadImage = new JPanel();
        addBorder(loadImage, "Load Image");

        JPanel training = new JPanel();
        addBorder(training, "Training");

        JPanel saveLoad = new JPanel();
        addBorder(saveLoad, "File");

        final JPanel drawing = new JPanel();
        addBorder(drawing, "Drawing");

        nextImage = new JButton("Next image");
        train = new JButton("Train");
        label = new JLabel("-");
        progress = new JLabel("");
        classify = new JButton("Classify");
        result = new JLabel("-");
        save = new JButton("Save weights");
        load = new JButton("Load weights");
        erase = new JButton("Erase");


        GroupLayout layout = new GroupLayout(loadImage);
        loadImage.setLayout(layout);
        layout.setHorizontalGroup(
                layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                                .addComponent(nextImage)
                                .addComponent(label))
        );
        layout.setVerticalGroup(
                layout.createSequentialGroup()
                        .addComponent(nextImage)
                        .addComponent(label)
        );

        GroupLayout layout2 = new GroupLayout(training);
        training.setLayout(layout2);
        layout2.setHorizontalGroup(
                layout2.createSequentialGroup()
                        .addGroup(layout2.createParallelGroup(GroupLayout.Alignment.LEADING)
                                .addComponent(train)
                                .addComponent(progress)
                                .addComponent(classify)
                                .addComponent(result)
                        )
        );
        layout2.setVerticalGroup(
                layout2.createSequentialGroup()
                        .addComponent(train)
                        .addComponent(progress)
                        .addComponent(classify)
                        .addComponent(result)
        );

        GroupLayout layout3 = new GroupLayout(saveLoad);
        saveLoad.setLayout(layout3);
        layout3.setHorizontalGroup(
                layout3.createSequentialGroup()
                        .addGroup(layout3.createParallelGroup(GroupLayout.Alignment.LEADING)
                                        .addComponent(save)
                                        .addComponent(load)
                        )
        );
        layout3.setVerticalGroup(
                layout3.createSequentialGroup()
                        .addComponent(save)
                        .addComponent(load)
        );

        GroupLayout layout4 = new GroupLayout(drawing);
        drawing.setLayout(layout4);
        layout4.setHorizontalGroup(
                layout4.createSequentialGroup()
                        .addGroup(layout4.createParallelGroup(GroupLayout.Alignment.LEADING)
                                        .addComponent(erase)
                        )
        );
        layout4.setVerticalGroup(
                layout4.createSequentialGroup()
                        .addComponent(erase)
        );


        GroupLayout overall = new GroupLayout(this);
        this.setLayout(overall);

        overall.setHorizontalGroup(
                overall.createSequentialGroup()
                        .addGroup(overall.createParallelGroup(GroupLayout.Alignment.LEADING)
                                .addComponent(loadImage)
                                .addComponent(training)
                                .addComponent(saveLoad)
                                .addComponent(drawing))
        );
        overall.setVerticalGroup(
                overall.createSequentialGroup()
                        .addComponent(loadImage)
                        .addComponent(training)
                        .addComponent(saveLoad)
                        .addComponent(drawing)
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
                DrawingPanel drawingPanel = DrawingPanel.getInstance();
                drawingPanel.display(image);
                updateLabel(image.getLabel());
            }
        });

        train.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                progress.setText("Training...");

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

                progress.setText("Net trained.");
            }
        });

        classify.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                DrawingPanel drawingPanel = DrawingPanel.getInstance();
                Image currentImage = drawingPanel.getImage();
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

        erase.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                DrawingPanel drawingPanel = DrawingPanel.getInstance();
                drawingPanel.resetImage();
            }
        });
    }

    public void updateLabel(String label) {
        if (this.label != null) {
            this.label.setText("Digit: " + label);
        }
    }

    private void addBorder(JComponent component, String title) {
        Border etch = BorderFactory.createEtchedBorder(EtchedBorder.LOWERED);
        Border tb = BorderFactory.createTitledBorder(etch,title);
        component.setBorder(tb);
    }
}


