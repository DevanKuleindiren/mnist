package com.devankuleindiren.mnist;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;

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
}
