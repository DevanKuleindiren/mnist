package com.devankuleindiren.mnist;

public class Batch {

    private double[][] inputs;
    private double[][] targets;

    public Batch (double[][] inputs, double[][] targets) {
        this.inputs = inputs;
        this.targets = targets;
    }

    public void setInputs (double[][] inputs) {
        this.inputs = inputs;
    }

    public void setTargets (double[][] targets) {
        this.targets = targets;
    }

    public double[][] getInputs() {
        return inputs;
    }

    public double[][] getTargets() {
        return targets;
    }
}
