package com.devankuleindiren.mnist;

import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.border.EtchedBorder;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;
import java.util.regex.Pattern;

public class ControlPanel extends JPanel {

    private static ControlPanel instance = null;

    private JTextField loadImageSource;
    private JButton nextImage;
    private JLabel label;

    private JTextField trainingSource;
    private JButton train;
    private JLabel progress;
    private JButton classify;
    private JLabel result;

    private JTextField weightsSource;
    private JButton saveWeights;
    private JButton load;

    private JButton erase;

    private JTextField artificialDataSource;
    private JButton generate;
    private JButton saveData;

    private int trainIter = 1000;
    private int batchSize = 1000;
    private double lR = 0.001;
    private double beta = 1.0;

    private int inputNodesNo = 785;
    private int hiddenNeuronNo = 15;
    private int outputNeuronNo = 12;

    public static ControlPanel getInstance() {
        if (instance == null) instance = new ControlPanel();

        return instance;
    }

    private ControlPanel() {
        super();

        // SECTIONS OF THE CONTROL PANEL

        final JPanel loadImage = new JPanel();
        addBorder(loadImage, "Load Image");

        JPanel training = new JPanel();
        addBorder(training, "Training");

        final JPanel saveLoad = new JPanel();
        addBorder(saveLoad, "File");

        JPanel drawing = new JPanel();
        addBorder(drawing, "Drawing");

        final JPanel artificialData = new JPanel();
        addBorder(artificialData, "Artificial Data");

        loadImageSource = new JTextField("mnist_test.csv");
        nextImage = new JButton("Next image");
        label = new JLabel("-");
        trainingSource = new JTextField("mnist_train.csv");
        train = new JButton("Train");
        progress = new JLabel("");
        classify = new JButton("Classify");
        result = new JLabel("-");
        weightsSource = new JTextField("weights.txt");
        saveWeights = new JButton("Save weights");
        load = new JButton("Load weights");
        erase = new JButton("Erase");
        artificialDataSource = new JTextField("artificialData.txt");
        generate = new JButton("Generate");
        saveData = new JButton("Save");

        GroupLayout layout = new GroupLayout(loadImage);
        loadImage.setLayout(layout);
        layout.setHorizontalGroup(
                layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                                .addComponent(loadImageSource)
                                .addComponent(nextImage)
                                .addComponent(label))
        );
        layout.setVerticalGroup(
                layout.createSequentialGroup()
                        .addComponent(loadImageSource)
                        .addComponent(nextImage)
                        .addComponent(label)
        );

        GroupLayout layout2 = new GroupLayout(training);
        training.setLayout(layout2);
        layout2.setHorizontalGroup(
                layout2.createSequentialGroup()
                        .addGroup(layout2.createParallelGroup(GroupLayout.Alignment.LEADING)
                                        .addComponent(trainingSource)
                                        .addComponent(train)
                                        .addComponent(progress)
                                        .addComponent(classify)
                                        .addComponent(result)
                        )
        );
        layout2.setVerticalGroup(
                layout2.createSequentialGroup()
                        .addComponent(trainingSource)
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
                                        .addComponent(weightsSource)
                                        .addComponent(saveWeights)
                                        .addComponent(load)
                        )
        );
        layout3.setVerticalGroup(
                layout3.createSequentialGroup()
                        .addComponent(weightsSource)
                        .addComponent(saveWeights)
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

        GroupLayout layout5 = new GroupLayout(artificialData);
        artificialData.setLayout(layout5);
        layout5.setHorizontalGroup(
                layout5.createSequentialGroup()
                        .addGroup(layout5.createParallelGroup(GroupLayout.Alignment.LEADING)
                                        .addComponent(artificialDataSource)
                                        .addComponent(generate)
                                        .addComponent(saveData)
                        )
        );
        layout5.setVerticalGroup(
                layout5.createSequentialGroup()
                        .addComponent(artificialDataSource)
                        .addComponent(generate)
                        .addComponent(saveData)
        );


        GroupLayout overall = new GroupLayout(this);
        this.setLayout(overall);

        overall.setHorizontalGroup(
                overall.createSequentialGroup()
                        .addGroup(overall.createParallelGroup(GroupLayout.Alignment.LEADING)
                                .addComponent(loadImage)
                                .addComponent(training)
                                .addComponent(saveLoad)
                                .addComponent(drawing)
                                .addComponent(artificialData))
        );
        overall.setVerticalGroup(
                overall.createSequentialGroup()
                        .addComponent(loadImage)
                        .addComponent(training)
                        .addComponent(saveLoad)
                        .addComponent(drawing)
                        .addComponent(artificialData)
        );


        nextImage.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                DataLoader dataLoader = DataLoader.getInstance();
                try {
                    Image image = dataLoader.next(loadImageSource.getText());
                    DrawingPanel drawingPanel = DrawingPanel.getInstance();
                    drawingPanel.display(image);
                    updateLabel(image.getLabel());
                } catch (IOException exception) {
                    JOptionPane.showMessageDialog(null, exception.getMessage());
                }
            }
        });

        train.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                progress.setText("Training...");

                // LOAD DATA FROM DATA LOADER
                System.out.println("LOADING INPUT BATCH...");
                try {
                    final Batch batch = DataLoader.getInputBatch(batchSize, trainingSource.getText());

                    System.out.println("INSTANTIATING DEEPNET...");
                    final DeepNet deepNet = DeepNet.getInstance(inputNodesNo, hiddenNeuronNo, outputNeuronNo);

                    System.out.println("TRAINING DEEPNET...");
                    double error = deepNet.trainNet(batch.getInputs(), batch.getTargets(), lR, beta, trainIter);
                    System.out.println();
                    System.out.println();
                    System.out.println("DEEPNET TRAINED. ERROR: " + Double.toString((error / (batchSize * 10)) * 100) + " %");

                    progress.setText("Net trained.");
                } catch (IOException exception) {
                    JOptionPane.showMessageDialog(null, exception.getMessage());
                }
            }
        });

        classify.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                classify();
            }
        });

        saveWeights.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(weightsSource.getText(), false));

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
                    JOptionPane.showMessageDialog(null, "Saved weights to " + weightsSource.getText());

                } catch (FileNotFoundException exception) {
                    JOptionPane.showMessageDialog(null, weightsSource.getText() + " could not be found.");
                } catch (IOException exception) {
                    JOptionPane.showMessageDialog(null, weightsSource.getText() + " could not be saved to.");
                }
            }
        });

        load.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                loadWeights();
            }
        });

        erase.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                DrawingPanel drawingPanel = DrawingPanel.getInstance();
                drawingPanel.resetImage();
            }
        });

        generate.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                ArtificialDataPanel artificialDataPanel = ArtificialDataPanel.getInstance();
                DrawingPanel drawingPanel = DrawingPanel.getInstance();
                artificialDataPanel.generateDataWithImage(drawingPanel.getImage());
            }
        });

        saveData.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    String predictedTarget = classify();
                    boolean targetValid = false;

                    String target = JOptionPane.showInputDialog("Is this the right character?", predictedTarget);
                    // Check target is an appropriate character
                    if (target != null && Pattern.matches("[0-9+-]", target)) {

                        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(artificialDataSource.getText(), true));
                        ArtificialDataPanel artificialDataPanel = ArtificialDataPanel.getInstance();

                        for (SnapShot s : artificialDataPanel.getSnapShots()) {
                            bufferedWriter.write(target);
                            for (int row = 0; row < s.getImage().getHeight(); row++) {
                                for (int col = 0; col < s.getImage().getWidth(); col++) {
                                    bufferedWriter.write("," + s.getImage().getPixel(row, col));
                                }
                            }
                            bufferedWriter.write("\n");
                        }
                        bufferedWriter.close();

                        JOptionPane.showMessageDialog(null, "Saved artificial data to " + artificialDataSource.getText());
                    } else {
                        JOptionPane.showMessageDialog(null, target + " is an invalid target value.");
                    }
                } catch (FileNotFoundException exception) {
                    JOptionPane.showMessageDialog(null, artificialDataSource.getText() + " could not be found.");
                } catch (IOException exception) {
                    JOptionPane.showMessageDialog(null, artificialDataSource.getText() + " could not be saved to.");
                }

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

    public void loadWeights () {
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(weightsSource.getText()));

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
                JOptionPane.showMessageDialog(null, "Loaded weights from " + weightsSource.getText());

            } else throw new InvalidWeightFormatException("The network of weights in the file is invalid.");

        } catch (FileNotFoundException exception) {
            JOptionPane.showMessageDialog(null, weightsSource.getText() + " could not be found.");
        } catch (IOException exception) {
            JOptionPane.showMessageDialog(null, weightsSource.getText() + " could not be loaded.");
        } catch (InvalidWeightFormatException exception) {
            JOptionPane.showMessageDialog(null, exception.getMessage());
        }
    }

    public String classify () {
        DrawingPanel drawingPanel = DrawingPanel.getInstance();
        Image currentImage = drawingPanel.getImage();
        double[][] input = currentImage.pixelsToVector();
        double[][] output;

        DeepNet deepNet = DeepNet.getInstance(inputNodesNo, hiddenNeuronNo, outputNeuronNo);

        output = deepNet.useNet(input, 1.0);
        output = deepNet.rectifyActivations(output);

        String target = "0";
        for (int i = 0; i < output[0].length; i++) {
            if (output[0][i] == 1) {
                if (i < 10) {
                    target = Integer.toString(i);
                } else if (i == 10) {
                    target = "+";
                } else {
                    target = "-";
                }
                result.setText("Result: " + target);
            }
        }
        return target;
    }
}


