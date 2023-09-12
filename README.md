Perceptron
==========

This repo is mainly a way for me to understand neural networks, backpropagation and some basics of machine learning.

To run in, just do:
```
go run cmd/main.go
```

Sample output:
```
$ go run cmd/main.go
Perceptron
==========

Training....
Final error on 1000 epochs: 4.002609

Predicting....
Accuracy on 210 predictions: 99.047619%
```

The main program will load a [dataset](http://archive.ics.uci.edu/dataset/236/seeds) from UC Irvine containing a classification of wheat seeds based on its geometrical properties. Then it will create a neural network with 3 "layers": 

* an Input layer with 7 nodes (one for each input property)
* a hidden layer with 5 nodes
* an output layer with 3 nodes (one for each possible class of seed)

Then the network will be trained with the given dataset, and finally we again use the same dataset to let the network "predict" the classification of each seed based on its inputs.

![Wheat Classification Neural Network](wheatomatic9000.png)

The network configuration can be chaned in `cmd/main.go` on this line:

```
    nn := neuralnet.New(7,5,3)
```
