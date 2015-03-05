package com.devankuleindiren.mnist;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;

public class ControlPanel extends JPanel {

    private static ControlPanel instance = null;

    private JButton nextImage;
    private JButton train;
    private JButton classify;
    private JProgressBar progressBar;
    private JLabel label;
    private JLabel result;

    private int trainIter = 1000;
    private int batchSize = 1000;
    private double lR = 0.001;
    private double beta = 1.0;

    public static ControlPanel getInstance () {
        if (instance == null) instance = new ControlPanel();

        return instance;
    }

    private ControlPanel() {
        super();

        nextImage = new JButton("Next image");
        train = new JButton("Train");
        classify = new JButton("Classify");
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
                                .addComponent(result))
        );
        layout.setVerticalGroup(
                layout.createSequentialGroup()
                        .addComponent(nextImage)
                        .addComponent(label)
                        .addComponent(train)
                        .addComponent(progressBar)
                        .addComponent(classify)
                        .addComponent(result)
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
                final DeepNet deepNet = DeepNet.getInstance(785, 15, 10);

                System.out.println("TRAINING DEEPNET...");
                double error = deepNet.trainNet(batch.getInputs(), batch.getTargets(), lR, beta, trainIter);
                System.out.println();
                System.out.println();
                System.out.println("DEEPNET TRAINED. ERROR: " + Double.toString((error / (batchSize * 10)) * 100));
            }
        });

        classify.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                GamePanel gamePanel = GamePanel.getInstance();
                Image currentImage = gamePanel.getImage();
                double[][] input = currentImage.pixelsToVector();
                double[][] output;

                DeepNet deepNet = DeepNet.getInstance(785, 15, 10);

                output = deepNet.useNet(input, 1.0);
                output = deepNet.rectifyActivations(output);

                for (int i = 0; i < output[0].length; i++) {
                    if (output[0][i] == 1) result.setText("Result: " + i);
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


