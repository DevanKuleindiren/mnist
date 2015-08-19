package com.devankuleindiren.mnist;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;

public class DrawingPanel extends JPanel implements MouseListener, MouseMotionListener {

    private static DrawingPanel instance = null;
    private int zoom = 15; //Number of pixels used to represent a cell
    private int width = 1; //Width of game board in pixels
    private int height = 1;//Height of game board in pixels
    private Image image = null;
    private boolean mouseDown = false;

    private DrawingPanel () {
        addMouseListener(this);
        addMouseMotionListener(this);
        this.setAlignmentX(JComponent.CENTER_ALIGNMENT);
        this.setAlignmentY(JComponent.CENTER_ALIGNMENT);

    }


    public Dimension getPreferredSize() {
        return new Dimension(width, height);
    }

    public static DrawingPanel getInstance () {
        if (instance == null) instance = new DrawingPanel();

        return instance;
    }

    public Image getImage () {
        return image;
    }

    protected void paintComponent(Graphics g) {
        if (image == null) return;
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, width, height);
        image.draw(g, width, height);
    }

    public void display(Image w) {
        image = w;
        int newWidth = image.getWidth() * zoom;
        int newHeight = image.getHeight() * zoom;
        if (newWidth != width || newHeight != height) {
            width = newWidth;
            height = newHeight;
            revalidate(); //trigger the DrawingPanel to re-layout its components
        }
        repaint();
    }

    public void resetImage () {

        for (int row = 0; row < image.getHeight(); row++) {
            for (int col = 0; col < image.getWidth(); col++) {
                image.setPixel(row, col, 0);
            }
        }
        repaint();
    }

    @Override
    public void mouseClicked(MouseEvent e) {
        updateImage(e.getX(), e.getY());
    }

    @Override
    public void mousePressed(MouseEvent e) {
        mouseDown = true;
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        mouseDown = false;
    }

    @Override
    public void mouseEntered(MouseEvent e) {

    }

    @Override
    public void mouseExited(MouseEvent e) {

    }

    @Override
    public void mouseDragged(MouseEvent e) {
        if (mouseDown) updateImage(e.getX(), e.getY());
    }

    @Override
    public void mouseMoved(MouseEvent e) {
        if (mouseDown) updateImage(e.getX(), e.getY());
    }

    private void updateImage (int x, int y) {
        if (image != null) {
            int row = y / zoom;
            int col = x / zoom;

            int radius = 2;

            for (int r = row - radius; r <= row + radius; r++) {
                for (int c = col - radius; c <= col + radius; c++) {
                    int modR2 = (int) Math.pow(r - row,2);
                    int modC2 = (int) Math.pow(c - col,2);
                    double mod = Math.sqrt(modR2 + modC2);

                    int val = (int) (255 - ((255.0 / (double) radius) * mod));
                    if (val < 0) val = 0;

                    image.setPixel(r, c, image.getPixel(r, c) + val);
                }
            }

            repaint();
        }
    }
}
