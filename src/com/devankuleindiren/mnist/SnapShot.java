package com.devankuleindiren.mnist;

import javax.swing.*;
import java.awt.*;

/**
 * Created by Devan Kuleindiren on 17/08/15.
 */
public class SnapShot extends JPanel {

    private Image image;
    private int width;
    private int height;
    private int zoom = 3;

    public void update(Image image) {
        this.image = image;
        repaint();
    }

    public Dimension getPreferredSize() {
        return new Dimension(image.getWidth() * zoom, image.getHeight() * zoom);
    }

    public SnapShot(Image image) {
        super();
        this.image = image;
        this.width = image.getWidth() * zoom;
        this.height = image.getHeight() * zoom;
        repaint();
    }

    protected void paintComponent(Graphics g) {
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, width, height);
        image.draw(g, width, height);
        g.setColor(Color.LIGHT_GRAY);
    }

    public Image getImage () {
        return image;
    }
}
