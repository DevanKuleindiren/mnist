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
        int maxPinShift = 8;

        snapShots.get(0).update(image);

        for (int i = 1; i < snapShots.size(); i++) {

            Image clone = (Image) image.clone();

            // Curve with Pin shift
            if (random.nextInt(2) == 0) {
                double horizontalCurve = (random.nextDouble() - 0.5) * 3;
                int horizontalPinShift = random.nextInt((maxPinShift * 2) + 1) - maxPinShift;
                clone.curveHorizontal(horizontalCurve, horizontalPinShift);
            } else {
                double verticalCurve = (random.nextDouble() - 0.5) * 3;
                int verticalPinShift = random.nextInt((maxPinShift * 2) + 1) - maxPinShift;
                clone.curveVertical(verticalCurve, verticalPinShift);
            }

            // Shift
            int horizontalShift = random.nextInt((maxShift * 2) + 1) - maxShift;
            int verticalShift = random.nextInt((maxShift * 2) + 1) - maxShift;
            clone.shiftHorizontal(horizontalShift);
            clone.shiftVertical(verticalShift);

            // Scale
            double horizontalStretch = random.nextDouble() - 0.4;
            clone.stretchHorizontal(clone.getWidth() / 2, horizontalStretch);

            double verticalStretch = random.nextDouble() - 0.4;
            clone.stretchVertical(clone.getWidth() / 2, verticalStretch);

            snapShots.get(i).update(clone);
        }
    }

    public ArrayList<SnapShot> getSnapShots () {
        return snapShots;
    }
}
