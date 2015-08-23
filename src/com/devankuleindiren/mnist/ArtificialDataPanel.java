package com.devankuleindiren.mnist;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.Random;

/**
 * Created by Devan Kuleindiren on 17/08/15.
 */
public class ArtificialDataPanel extends JPanel {

    public static ArtificialDataPanel instance = null;
    private ArrayList<SnapShot> snapShots;

    public static ArtificialDataPanel getInstance() {
        if (instance == null) {
            instance = new ArtificialDataPanel();
        }
        return instance;
    }

    public ArtificialDataPanel () {
        super();
        setLayout(new BorderLayout());

        snapShots = new ArrayList<SnapShot>();

        for (int i = 0; i < 10; i++) {
            snapShots.add(new SnapShot(new Image(28, 28)));
        }

        GroupLayout layout = new GroupLayout(this);
        this.setLayout(layout);

        layout.setAutoCreateGaps(true);
        layout.setAutoCreateContainerGaps(true);

        GroupLayout.ParallelGroup parallelGroup = layout.createParallelGroup(GroupLayout.Alignment.LEADING);
        GroupLayout.SequentialGroup sequentialGroup = layout.createSequentialGroup();
        for (SnapShot s : snapShots) {
            parallelGroup.addComponent(s);
            sequentialGroup.addComponent(s);
        }

        layout.setVerticalGroup(
                layout.createSequentialGroup()
                        .addGroup(parallelGroup));
        layout.setHorizontalGroup(sequentialGroup);
    }

    public void generateDataWithImage (Image image) {
        Random random = new Random();
        int maxShift = 2;
        int maxBlurRadius = 1;
        int maxPinShift = 8;

        for (SnapShot s : snapShots) {

            Image clone = (Image) image.clone();

            // Curve with Pin shift
            if (random.nextInt(2) == 0) {
                double horizontalCurve = (random.nextDouble() - 0.5) * 10;
                int horizontalPinShift = random.nextInt((maxPinShift * 2) + 1) - maxPinShift;
                clone.curveHorizontal(horizontalCurve, horizontalPinShift);
            }
            if (random.nextInt(2) == 0) {
                double verticalCurve = (random.nextDouble() - 0.5) * 10;
                int verticalPinShift = random.nextInt((maxPinShift * 2) + 1) - maxPinShift;
                clone.curveVertical(verticalCurve, verticalPinShift);
            }

            // Shift
            int horizontalShift = random.nextInt((maxShift * 2) + 1) - maxShift;
            int verticalShift = random.nextInt((maxShift * 2) + 1) - maxShift;
            clone.shiftHorizontal(horizontalShift);
            clone.shiftVertical(verticalShift);

//            // Blur
//            double blur = (random.nextDouble() / 20.0) + 0.15;
//            int radius = random.nextInt(maxBlurRadius + 1);
//            clone.blur(blur, radius);

            // Scale
            if (random.nextInt(5) == 0) {
                double horizontalStretch = (random.nextDouble() / 4.0) + 1.0;
                clone.stretchHorizontal(clone.getWidth() / 2, horizontalStretch);
            }
            if (random.nextInt(5) == 0) {
                double verticalStretch = (random.nextDouble() / 4.0) + 1.0;
                clone.stretchVertical(clone.getWidth() / 2, verticalStretch);
            }

            s.update(clone);
        }
    }
}
