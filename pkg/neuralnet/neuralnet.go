package neuralnet

import (
	"fmt"
    "math"
    "math/rand"
    "github.com/oarevalo/perceptron/pkg/perceptron"
)

type NeuralNet struct {
	in []*perceptron.Perceptron
	out []*perceptron.Perceptron
}

// Creates a new Neural Network with an input layer, a hidden layer 
// and an output layer. Numbers of nodes on each layer are given as 
// arguments
func New(num_in, num_hidden, num_out int) *NeuralNet {
	in, out := create_network(num_in, num_hidden, num_out, sigmoid)
	nn := &NeuralNet{}
	nn.in = in
	nn.out = out
	return nn
}

// Runs a single training iteration of the network with the given inputs,
// expected output and learning rate. Returns the squared sum of the errors.
// Training is done by forward propagating the inputs, calculating and
// backpropagating errors and then adjusting the weights on all nodes
func (nn *NeuralNet) Train(rate float64, inputs []float64, output int) float64 {
    // set inputs
    for k,v := range inputs {
        nn.in[k].Sense(v)
    }

    // forward propagate
    for k,_ := range inputs {
        nn.in[k].Propagate()
    }

    // calculate error
    sum_error := 0.0
    expected := make([]float64, len(nn.out))
    expected[output] = 1
    for k,v := range expected {
        sum_error = sum_error + math.Pow((expected[k]-nn.out[k].Value),2)
        nn.out[k].Expected(v)
    }

    // backpropagate
    for k,_ := range expected {
        nn.out[k].Backpropagate()
    }

    // update weights based on error
    for k,_ := range inputs {
        nn.in[k].Train(rate)
    }

    return sum_error
}

// Uses the trained neural network to predict the output for a 
// given set of inputs
func (nn *NeuralNet) Predict(inputs []float64) int {
    // set inputs
    for k,v := range inputs {
        nn.in[k].Sense(v)
    }

    // forward propagate
    for k,_ := range inputs {
        nn.in[k].Propagate()
    }

    // get predicted value (via arg max function)
    return argMax(nn.out)
}


// Sigmoid function for use on neuron transfer/activation
func sigmoid(input float64) (output float64) {
    output = 1.0 / (1.0 + math.Exp(-input))
    return
}

// argMax returns the index position of the element in the
// slice that has the greater value
func argMax(layer []*perceptron.Perceptron) int {
    // create a slice of all the values on the layer
    // and get the index of the max value
    max := math.MaxFloat64 * -1
    pos := -1
    for i,p := range layer {
        if p.Value > max {
            max = p.Value
            pos = i
        }
    }
    return pos
}

// creates and initializes a set of connected neurons with the given
// configuration
func create_network(num_in, num_hidden, num_out int, fn perceptron.ActFn) ([]*perceptron.Perceptron, []*perceptron.Perceptron) {
    p_in := make([]*perceptron.Perceptron, num_in)
    p_h := make([]*perceptron.Perceptron, num_hidden)
    p_out := make([]*perceptron.Perceptron, num_out)

    // input layer
    for i := 0; i < num_in; i++ {
        p := perceptron.Sensor(fmt.Sprintf("input_%d",i))
        p_in[i] = p
    }

    // hidden layer
    for i := 0; i < num_hidden; i++ {
        b := rand.Float64()     // initialize to random bias
        p := perceptron.Neuron(fmt.Sprintf("hidden_%d",i), fn, b)
        for _,in := range p_in {
            w := rand.Float64()
            p.AddInput(in, w)
            in.AddOutput(p, w)
        }
        p_h[i] = p
    }

    // output layer
    for i := 0; i < num_out; i++ {
        b := rand.Float64()     // initialize to random bias
        p := perceptron.Neuron(fmt.Sprintf("output_%d",i), fn, b)
        for _,h := range p_h {
            w := rand.Float64()
            p.AddInput(h, w)
            h.AddOutput(p, w)
        }
        p_out[i] = p
    }

    return p_in, p_out
}