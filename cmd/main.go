package main

import (
    "bufio"
    "fmt"
    "os"
    "math"
    "strings"
    "strconv"
    "github.com/oarevalo/perceptron/pkg/neuralnet"
)

func main() {
    fmt.Println("Perceptron")
    fmt.Println("==========\n")

    // create a new neural net with 3 layers (inputs/hidden/outputs)
    nn := neuralnet.New(7,5,3)

    // load a dataset to use for training and evaluation
    // and normalize all values to be between 0 and 1
    // (needed because the NN uses the Sigmoid function as transfer)
    dataset := readDataFile("data/seeds_dataset.txt")
    minmax:=minMax(dataset)
    dataset=normalizeDataset(dataset,minmax)

    // Training
    fmt.Println("Training....")
    epochs := 1000
    learning_rate := 0.3
    error := 0.0
    for i := 0; i < epochs; i++ {
        sum_error := 0.0
        for _,row := range dataset {
            inputs := row[:len(row)-1]
            output := int(row[len(row)-1])-1
            sum_error = sum_error + nn.Train(learning_rate, inputs, output)
        }
        error = sum_error
    }
    fmt.Printf("Final error on %d epochs: %f\n",epochs,error)

    // Prediction
    fmt.Println("\nPredicting....")
    correct := 0
    for _,row := range dataset {
        inputs := row[:len(row)-1]
        output := int(row[len(row)-1])-1
        prediction := nn.Predict(inputs)
        if int(prediction) == output {
            correct = correct + 1
        }
    }
    fmt.Printf("Accuracy on %d predictions: %.2f%%\n",len(dataset), (float64(correct)/float64(len(dataset)))*100)
}

func readDataFile(filepath string) [][]float64 {
    dataset := [][]float64{}

    // open file
    f, err := os.Open(filepath)
    if err != nil {
        panic(err)
    }
    defer f.Close()

    // read file
    scanner := bufio.NewScanner(f)
    for scanner.Scan() {
        line := scanner.Text()
        fields := strings.Fields(line)
        row := []float64{}
        for _,v := range fields {
            value, _ := strconv.ParseFloat(v,64)
            row = append(row, value)
        }
        dataset = append(dataset, row)
    }

    return dataset
}

func minMax(dataset [][]float64) [][]float64 {
    stats := [][]float64{}

    for i,row := range dataset {
        if i == 0 {
            for j:=0; j<len(row); j++ {
                stats = append(stats, []float64{math.MaxFloat64, math.MaxFloat64*-1})
            }        
        }
        for j,value := range row {
            if value < stats[j][0] {
                stats[j][0] = value
            }
            if value > stats[j][1] {
                stats[j][1] = value
            }
        }
    }

    return stats
}

func normalizeDataset(dataset [][]float64, minmax [][]float64) [][]float64 {
    for i,row := range dataset {
        for j,_ := range row[:len(row)-1] {
            dataset[i][j] = (dataset[i][j] - minmax[j][0]) / (minmax[j][1] - minmax[j][0])
        }
    }
    return dataset
}