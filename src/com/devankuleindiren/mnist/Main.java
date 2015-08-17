package com.devankuleindiren.mnist;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.border.EtchedBorder;

public class Main extends JFrame {

    private DrawingPanel DrawingPanel;
    private DataLoader dataLoader;
    private ControlPanel controlPanel;

    public Main() {
        super("MNIST");
        dataLoader = DataLoader.getInstance();
        setSize(640, 480);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        JComponent DrawingPanel = createDrawingPanel();
        add(DrawingPanel, BorderLayout.CENTER);
        JComponent controlPanel = createControlPanel();
        add(controlPanel, BorderLayout.WEST);
    }

    private void addBorder(JComponent component, String title) {
        Border etch = BorderFactory.createEtchedBorder(EtchedBorder.LOWERED);
        Border tb = BorderFactory.createTitledBorder(etch,title);
        component.setBorder(tb);
    }

    private JComponent createDrawingPanel() {
        JPanel holder = new JPanel();
        //holder.setLayout(new BoxLayout(holder, BoxLayout.X_AXIS));
        addBorder(holder, "Image");

        final DrawingPanel result = DrawingPanel.getInstance();
        JButton reset = new JButton("Reset");
        reset.setAlignmentX(JComponent.CENTER_ALIGNMENT);
        reset.setAlignmentY(JComponent.CENTER_ALIGNMENT);
        reset.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                result.resetImage();
            }
        });

        holder.add(result);
        holder.add(reset);

        this.DrawingPanel = result;
        return new JScrollPane(holder);
    }

    private JComponent createControlPanel() {
        JPanel holder = new JPanel();
        addBorder(holder, "Control");
        ControlPanel result = ControlPanel.getInstance();
        holder.add(result);
        this.controlPanel = result;
        return holder;
    }

    private void resetWorld() {

        Image image = null;
        try {
            image = dataLoader.next();
            controlPanel.updateLabel(image.getLabel());
        } catch (IOException e) {
            JOptionPane.showMessageDialog(this,
                    "Error loading data",
                    "An error occurred when importing the data. " + e.getMessage(),
                    JOptionPane.ERROR_MESSAGE);
        }
        DrawingPanel.display(image);
        repaint();
    }

    public static void main(String[] args) {
        Main gui = new Main();
        gui.resetWorld();
        gui.setVisible(true);
    }
}