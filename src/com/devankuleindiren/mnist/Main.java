package com.devankuleindiren.mnist;

import java.awt.BorderLayout;
import java.io.FileNotFoundException;
import java.io.IOException;
import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.border.EtchedBorder;

public class Main extends JFrame {

    private DataLoader dataLoader;
    private DrawingPanel drawingPanel;
    private ControlPanel controlPanel;
    private ArtificialDataPanel artificialDataPanel;

    public Main() {
        super("MNIST");
        dataLoader = DataLoader.getInstance();
        setSize(1200, 780);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        JComponent DrawingPanel = createDrawingPanel();
        add(DrawingPanel, BorderLayout.CENTER);
        JComponent controlPanel = createControlPanel();
        add(controlPanel, BorderLayout.WEST);
        JComponent artificialDataPanel = createArtificialDataPanel();
        add(artificialDataPanel, BorderLayout.SOUTH);
    }

    private void addBorder(JComponent component, String title) {
        Border etch = BorderFactory.createEtchedBorder(EtchedBorder.LOWERED);
        Border tb = BorderFactory.createTitledBorder(etch, title);
        component.setBorder(tb);
    }

    private JComponent createDrawingPanel() {
        JPanel holder = new JPanel();
        addBorder(holder, "Image");

        final DrawingPanel result = DrawingPanel.getInstance();
        holder.add(result);
        this.drawingPanel = result;
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

    private JComponent createArtificialDataPanel() {
        JPanel holder = new JPanel();
        addBorder(holder, "Artificial Data");
        ArtificialDataPanel result = ArtificialDataPanel.getInstance();
        this.artificialDataPanel = result;
        holder.add(result);
        JScrollPane scrollPane = new JScrollPane(holder);
        scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_NEVER);
        scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
        return scrollPane;
    }

    private void setUp() {

        Image image = null;
        try {
            image = dataLoader.next("mnist_test.csv");
            controlPanel.updateLabel(image.getLabel());
        } catch (IOException e) {
            JOptionPane.showMessageDialog(this,
                    e.getMessage(),
                    "Whoops!",
                    JOptionPane.ERROR_MESSAGE);
        }
        if (image == null) {
            image = new Image(28, 28);
        }
        drawingPanel.display(image);
        repaint();
    }

    public static void main(String[] args) {
        Main gui = new Main();
        gui.setUp();
        gui.setVisible(true);
    }
}