package neuralnet

import (
	"testing"
)

func TestNew(t *testing.T) {
	New(2,3,4)
}

func TestTrain(t *testing.T) {
	nn := New(2,2,2)

	dataset := getDataset()
	learning_rate := 0.1
    sum_error := 0.0

    for _,row := range dataset {
	    inputs := row[:len(row)-1]
	    output := int(row[len(row)-1])
        sum_error = sum_error + nn.Train(learning_rate, inputs, output)
	}

    t.Logf("Error: %f\n", sum_error)
}

func TestPredict(t *testing.T) {
	nn := New(2,2,2)

	dataset := getDataset()
    correct := 0

    for _,row := range dataset {
	    inputs := row[:len(row)-1]
	    output := int(row[len(row)-1])
        prediction := nn.Predict(inputs)
        if int(prediction) == output {
            correct = correct + 1
        }
   	}

    t.Logf("Accuracy on %d predictions: %.2f%%\n",len(dataset), (float64(correct)/float64(len(dataset)))*100)
}

func getDataset() [][]float64 {
    return [][]float64{
        {2.7810836,2.550537003,0},
        {1.465489372,2.362125076,0},
        {3.396561688,4.400293529,0},
        {1.38807019,1.850220317,0},
        {3.06407232,3.005305973,0},
        {7.627531214,2.759262235,1},
        {5.332441248,2.088626775,1},
        {6.922596716,1.77106367,1}, 
        {8.675418651,-0.242068655,1},
        {7.673756466,3.508563011,1},
    }
}