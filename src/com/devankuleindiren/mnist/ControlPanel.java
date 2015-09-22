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

    private static NeuralNetwork neuralNetwork;

    private JTextField loadImageSource;
    private JButton nextImage;
    private JLabel label;

    private JComboBox neuralNetworkDropdown;
    private JLabel trainingSourceLabel;
    private JTextField trainingSource;
    private JLabel batchSizeLabel;
    private JSpinner batchSizeSpinner;
    private JButton train;
    private JButton classify;
    private JLabel result;
    private JLabel trainingProgressLabel;
    private JProgressBar trainingProgressBar;

    private JTextField weightsSource;
    private JButton saveWeights;
    private JButton load;

    private JButton erase;

    private JTextField artificialDataSource;
    private JButton generate;
    private JButton saveData;

    private int inputNodesNo = 785;
    private int hiddenNeuronNo = 20;
    private int outputNeuronNo = 14;

    public static ControlPanel getInstance() {
        if (instance == null) instance = new ControlPanel();

        return instance;
    }

    private ControlPanel() {
        super();

        // GET INSTANCE OF NEURAL NETWORK
        neuralNetwork = new FNN2Layer (inputNodesNo, hiddenNeuronNo, outputNeuronNo);

        // SECTIONS OF THE CONTROL PANEL
        final JPanel loadImagePanel = new JPanel();
        addBorder(loadImagePanel, Strings.CONTROLPANEL_PANEL_LOADIMAGE);

        JPanel neuralNetworkPanel = new JPanel();
        addBorder(neuralNetworkPanel, Strings.CONTROLPANEL_PANEL_NEURALNETWORK);

        final JPanel saveLoadPanel = new JPanel();
        addBorder(saveLoadPanel, Strings.CONTROLPANEL_PANEL_FILE);

        JPanel drawingPanel = new JPanel();
        addBorder(drawingPanel, Strings.CONTROLPANEL_PANEL_DRAWING);

        final JPanel artificialDataPanel = new JPanel();
        addBorder(artificialDataPanel, Strings.CONTROLPANEL_PANEL_ARTIFICIALDATA);

        loadImageSource = new JTextField(Strings.CONTROLPANEL_SOURCES_TESTING);
        nextImage = new JButton(Strings.CONTROLPANEL_BUTTONS_NEXT);
        label = new JLabel("-");
        String[] neuralNetworkNames = { Strings.CONTROLPANEL_COMBOBOX_CNN, Strings.CONTROLPANEL_COMBOBOX_FNN };
        neuralNetworkDropdown = new JComboBox(neuralNetworkNames);
        neuralNetworkDropdown.setSelectedIndex(1);
        trainingSourceLabel = new JLabel(Strings.CONTROLPANEL_LABELS_TRAININGSOURCE);
        trainingSource = new JTextField(Strings.CONTROLPANEL_SOURCES_TRAINING);
        batchSizeLabel = new JLabel(Strings.CONTROLPANEL_LABELS_INPUTBATCHSIZE);
        batchSizeSpinner = new JSpinner();
        batchSizeSpinner.setModel(new SpinnerNumberModel(14000, 1, 100000, 1));
        train = new JButton(Strings.CONTROLPANEL_BUTTONS_TRAIN);
        classify = new JButton(Strings.CONTROLPANEL_BUTTONS_CLASSIFY);
        result = new JLabel("");
        trainingProgressLabel = new JLabel("");
        trainingProgressBar = new JProgressBar(0, 100);
        trainingProgressBar.setStringPainted(true);
        weightsSource = new JTextField(Strings.CONTROLPANEL_SOURCES_WEIGHTS);
        saveWeights = new JButton(Strings.CONTROLPANEL_BUTTONS_SAVEWEIGHTS);
        load = new JButton(Strings.CONTROLPANEL_BUTTONS_LOADWEIGHTS);
        erase = new JButton(Strings.CONTROLPANEL_BUTTONS_ERASE);
        artificialDataSource = new JTextField(Strings.CONTROLPANEL_SOURCES_ARTIFICIALDATA);
        generate = new JButton(Strings.CONTROLPANEL_BUTTONS_GENERATE);
        saveData = new JButton(Strings.CONTROLPANEL_BUTTONS_SAVEARTIFICIALDATA);

        GroupLayout layout = new GroupLayout(loadImagePanel);
        loadImagePanel.setLayout(layout);
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

        GroupLayout layout2 = new GroupLayout(neuralNetworkPanel);
        neuralNetworkPanel.setLayout(layout2);
        layout2.setHorizontalGroup(
                layout2.createSequentialGroup()
                        .addGroup(layout2.createParallelGroup(GroupLayout.Alignment.LEADING)
                                        .addComponent(neuralNetworkDropdown)
                                        .addComponent(trainingSourceLabel)
                                        .addComponent(trainingSource)
                                        .addComponent(batchSizeLabel)
                                        .addComponent(batchSizeSpinner)
                                        .addComponent(train)
                                        .addComponent(classify)
                                        .addComponent(result)
                                        .addComponent(trainingProgressLabel)
                                        .addComponent(trainingProgressBar)
                        )
        );
        layout2.setVerticalGroup(
                layout2.createSequentialGroup()
                        .addComponent(neuralNetworkDropdown)
                        .addComponent(trainingSourceLabel)
                        .addComponent(trainingSource)
                        .addComponent(batchSizeLabel)
                        .addComponent(batchSizeSpinner)
                        .addComponent(train)
                        .addComponent(classify)
                        .addComponent(result)
                        .addComponent(trainingProgressLabel)
                        .addComponent(trainingProgressBar)
        );

        GroupLayout layout3 = new GroupLayout(saveLoadPanel);
        saveLoadPanel.setLayout(layout3);
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

        GroupLayout layout4 = new GroupLayout(drawingPanel);
        drawingPanel.setLayout(layout4);
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

        GroupLayout layout5 = new GroupLayout(artificialDataPanel);
        artificialDataPanel.setLayout(layout5);
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
                                .addComponent(loadImagePanel)
                                .addComponent(neuralNetworkPanel)
                                .addComponent(saveLoadPanel)
                                .addComponent(drawingPanel)
                                .addComponent(artificialDataPanel))
        );
        overall.setVerticalGroup(
                overall.createSequentialGroup()
                        .addComponent(loadImagePanel)
                        .addComponent(neuralNetworkPanel)
                        .addComponent(saveLoadPanel)
                        .addComponent(drawingPanel)
                        .addComponent(artificialDataPanel)
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

        neuralNetworkDropdown.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JComboBox comboBox = (JComboBox) e.getSource();
                String selectedString = (String) comboBox.getSelectedItem();
                if (selectedString.equals(Strings.CONTROLPANEL_COMBOBOX_CNN)) {
                    if (!(neuralNetwork instanceof CNN_28x28)) {
                        neuralNetwork = new CNN_28x28();
                        loadWeights();
                    }
                } else {
                    if (!(neuralNetwork instanceof FNN2Layer)) {
                        neuralNetwork = new FNN2Layer(inputNodesNo, hiddenNeuronNo, outputNeuronNo);
                        loadWeights();
                    }
                }
            }
        });


        train.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

                // LOAD DATA FROM DATA LOADER
                System.out.println("LOADING INPUT BATCH...");
                try {
                    int batchSize = (Integer) batchSizeSpinner.getValue();
                    System.out.println(batchSize);

                    final MatrixBatch batch = DataLoader.getMatrixInputBatch(batchSize, trainingSource.getText());

                    System.out.println("TRAINING FNN...");
                    double error = neuralNetwork.train(batch.getInputs(), batch.getTargets());
                    System.out.println();
                    System.out.println();
                    System.out.println("FNN TRAINED. ERROR: " + Double.toString((error / (batchSize * 10)) * 100) + " %");

                } catch (IOException exception) {
                    JOptionPane.showMessageDialog(null, exception.getMessage());
                } catch (MatrixDimensionMismatchException exception) {
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
                    neuralNetwork.saveWeights(weightsSource.getText());
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
                    if (target != null && Pattern.matches("[0-9+-dx]", target)) {

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

    public void updateTrainingProgressBar(int progress, String status) {
        trainingProgressBar.setValue(progress);
        trainingProgressBar.setString(progress + "%");
        trainingProgressLabel.setText(status);

        if (status.equals(Strings.CONTROLPANEL_NEURALNETWORK_TRAININGINPROGRESS)) {
            train.setEnabled(false);
            classify.setEnabled(false);
        } else if (status.equals(Strings.CONTROLPANEL_NEURALNETWORK_TRAININGCOMPLETE)) {
            train.setEnabled(true);
            classify.setEnabled(true);
        }
    }

    private void addBorder(JComponent component, String title) {
        Border etch = BorderFactory.createEtchedBorder(EtchedBorder.LOWERED);
        Border tb = BorderFactory.createTitledBorder(etch,title);
        component.setBorder(tb);
    }

    public void loadWeights () {
        try {
            neuralNetwork.loadWeights(weightsSource.getText());
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
        Matrix input = currentImage.pixelsToVector();
        Matrix output;

        try {
            output = neuralNetwork.feedForward(input);
            output.rectifyActivations();

            String target = "0";
            for (int i = 0; i < output.getWidth(); i++) {
                if (output.get(0, i) == 1) {
                    if (i < 10) {
                        target = Integer.toString(i);
                    } else if (i == 10) {
                        target = "+";
                    } else if (i == 11) {
                        target = "-";
                    } else if (i == 12) {
                        target = "x";
                    } else {
                        target = "÷";
                    }
                    result.setText("Result: " + target);
                }
            }
            return target;
        } catch (MatrixDimensionMismatchException e) {
            JOptionPane.showMessageDialog(null, e.getMessage());
        }
        return "";
    }
}


