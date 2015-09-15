#!/bin/bash

cd src
cp ../out/production/MNIST/com/devankuleindiren/mnist/* com/devankuleindiren/mnist
jar -cfe MNIST.jar com.devankuleindiren.mnist.Main com/devankuleindiren/mnist/*
rm com/devankuleindiren/mnist/*.class
mv MNIST.jar ../MNIST.jar
