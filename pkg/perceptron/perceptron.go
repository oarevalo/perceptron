package perceptron

import (
    "fmt"
   // "math/rand"
)

type Perceptron struct {
    bias float64
    fn  ActFn
    inputs []*Synapse
    outputs []*Synapse

    Name string
    Type string
    Value float64
    Error float64
}

type Synapse struct {
    p *Perceptron
    w float64
}

// the signature of the activation function
type ActFn func(float64) float64


// creates a new perceptron
func Neuron(name string, actfn ActFn, b float64) *Perceptron {
    p := &Perceptron{}
    p.Type = "Neuron"
    p.Name = name
    p.fn = actfn
    p.bias = b
    return p
}

func Sensor(name string) *Perceptron {
    p := &Perceptron{}
    p.Type = "Sensor"
    p.Name = name
    return p
}

// adds input and outputs to the perceptron
func (p *Perceptron) AddInput(i *Perceptron, w float64) {
    p.inputs = append(p.inputs, &Synapse{i,w})
}
func (p *Perceptron) AddOutput(o *Perceptron, w float64) {
    p.outputs = append(p.outputs, &Synapse{o,w})
}
func (p *Perceptron) SetOutputWeight(o *Perceptron, w float64) {
    for _,out := range p.outputs {
        if out.p == o {
            out.w = w
        }
    }
}

// Neuron activation transforms all input given into a single 
// number (value/output of the neuron)
func (p *Perceptron) Activate() {
    activation := p.bias
    for _,in := range p.inputs {
        activation = activation + in.p.Value * in.w
    }
    p.Value = p.fn(activation)
}

func (p *Perceptron) Sense(x float64) {
    p.Value = x
}




// Propagate activates the current perceptron based on the value of its input 
// neurons and then causes the activation of each of its output neurons (if any)
func (p *Perceptron) Propagate() {
    // if this perceptron has input neurons then we need to activate it based on
    // the values of those neurons
    if p.Type == "Neuron" {
        p.Activate()
    }

    // now we need to propagate the value to the output neurons so that they can be activated
    for _,out := range p.outputs {
        out.p.Propagate()
    }
}

// this is where the perceptron evaluates the inputs and produces an output
// this method starts on an output neuron and lets input flow upstream until it
// reaches the beginning of the network (sensors)
func (p *Perceptron) Ingest(x []float64) {
    for i,in := range p.inputs {
        if in.p.Type == "Neuron" {
            in.p.Ingest(x)
        }
        if in.p.Type == "Sensor" {
            in.p.Sense(x[i])
        }
    }
    p.Activate()
}





// Back-propagate error values (output layer must already be set with its own errors)
func (p *Perceptron) Backpropagate() {
    if len(p.outputs) > 0 {
        error := 0.0
        for _,out := range p.outputs {
            error = error + out.w  * out.p.Error
        }
        p.CalculateError(error)
    }

    for _,in := range p.inputs {
        in.p.Backpropagate()
    }
}

func (p *Perceptron) Train(rate float64) {
    if len(p.inputs) > 0 {
        for _,s := range p.inputs {
            s.w = s.w - rate * p.Error * s.p.Value
            s.p.SetOutputWeight(p, s.w)
        }
        p.bias = p.bias - rate * p.Error
    }
    for _,s := range p.outputs {
        s.p.Train(rate)
    }
}

func (p *Perceptron) Expected(expected float64) {
    p.Error = (p.Value - expected) * transfer_derivative(p.Value)
}

func (p *Perceptron) CalculateError(err float64) {
    p.Error = err * transfer_derivative(p.Value)
}

func transfer_derivative(output float64) float64 {
    return output * (1.0 - output)
}



func (p *Perceptron) Info() {
    fmt.Println("============")
    fmt.Printf("Name: %s (%s)\n", p.Name, p.Type)
    fmt.Printf("Value: %f\n", p.Value)
    fmt.Printf("Error: %f\n", p.Error)
    fmt.Printf("Bias: %f\n", p.bias)
    for _,in := range p.inputs {
        fmt.Printf("Receives from %s (w: %f)\n", in.p.Name, in.w)
    }
    for _,out := range p.outputs {
        fmt.Printf("Sends to %s (w: %f)\n", out.p.Name, out.w)
    }
    fmt.Println("============")
}

func (p *Perceptron) Tree() {
    p.Info()
    for _,out := range p.outputs {
        out.p.Tree()
    }
}