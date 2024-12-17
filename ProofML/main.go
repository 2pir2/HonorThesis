package main

import (
	"encoding/json"
	"fmt"
	"math/big"
	"os"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/std/math/cmp"
)

// ProveModelCircuit defines the circuit structure for the neural network
type ProveModelCircuit struct {
	Weights  [2][5][5]frontend.Variable `gnark:",private"` // Weights as 3D slices
	Biases   [2][5]frontend.Variable    `gnark:",private"` // Biases as 2D slices
	Inputs   [1][5]frontend.Variable    `gnark:",public"`  // Input vectors as a 2D slice
	Expected [1]frontend.Variable       `gnark:",private"` // Expected outputs as a 1D slice
}

func scaleDown(api frontend.API, value frontend.Variable) frontend.Variable {
	// Compute quotient = value / 1000
	quotient := api.Div(value, 1000)

	// Compute nearest multiple = quotient * 1000
	nearestMultiple := api.Mul(quotient, 1000)

	// Compute remainder = value - nearestMultiple
	remainder := api.Sub(value, nearestMultiple)

	// Check if remainder >= 500
	isAboveThreshold := cmp.IsLess(api, 500, remainder)

	// Calculate adjustments
	roundUp := api.Sub(1000, remainder) // Amount to add for rounding up
	roundDown := api.Neg(remainder)     // Amount to subtract for rounding down

	// Conditionally apply round up or round down
	adjustment := api.Select(isAboveThreshold, roundUp, roundDown)

	// Apply the adjustment and return the scaled value
	return api.Add(value, adjustment)
}

func (circuit *ProveModelCircuit) Define(api frontend.API) error {
	largeBoundary := frontend.Variable(1000000000)

	// Iterate over each input vector
	for k := 0; k < len(circuit.Inputs); k++ {
		layerOutputs := circuit.Inputs[k] // Start with the input vector

		// Iterate over each layer
		for layer := 0; layer < len(circuit.Weights); layer++ {
			// Create a new slice for the outputs
			newOutputs := make([]frontend.Variable, len(circuit.Weights[layer]))

			// Iterate over each neuron in the layer
			for i := 0; i < len(circuit.Weights[layer]); i++ {
				sum := circuit.Biases[layer][i]

				// Compute the weighted sum
				for j := 0; j < len(circuit.Weights[layer][i]); j++ {
					tmp := api.Mul(circuit.Weights[layer][i][j], layerOutputs[j])
					sum = api.Add(sum, tmp)
				}

				// Apply the scale-down function
				//sum = scaleDown(api, sum)

				// Apply ReLU activation
				newOutputs[i] = applyReLU(api, sum, largeBoundary)
			}

			// Update layerOutputs by iterating through elements
			for i := range newOutputs {
				layerOutputs[i] = newOutputs[i]
			}

			// Print the outputs after the current layer
			api.Println("Outputs after layer", layer, ":", layerOutputs)
		}

		// Find the argmax in the final layer's output
		maxVal := layerOutputs[0]
		maxIdx := frontend.Variable(0)
		for i := 1; i < len(layerOutputs); i++ {
			isLess := cmp.IsLess(api, maxVal, layerOutputs[i])
			maxVal = api.Select(isLess, layerOutputs[i], maxVal)
			maxIdx = api.Select(isLess, frontend.Variable(i), maxIdx)
		}

		// Assert the predicted output matches the expected output
		api.AssertIsEqual(maxIdx, circuit.Expected[k])
	}

	return nil
}

// Apply ReLU activation function with scaling
func applyReLU(api frontend.API, value, largeBoundary frontend.Variable) frontend.Variable {
	isNegative := cmp.IsLess(api, value, 0)
	isLarge := cmp.IsLess(api, largeBoundary, value)
	return api.Select(api.Or(isNegative, isLarge), frontend.Variable(0), value)
}

func main() {
	// Open the JSON files
	weightsFile, err := os.Open("weights.json")
	if err != nil {
		fmt.Println("Error opening weights file:", err)
		return
	}
	defer weightsFile.Close()

	inputsFile, err := os.Open("inputs.json")
	if err != nil {
		fmt.Println("Error opening inputs file:", err)
		return
	}
	defer inputsFile.Close()

	outputsFile, err := os.Open("outputs.json")
	if err != nil {
		fmt.Println("Error opening outputs file:", err)
		return
	}
	defer outputsFile.Close()

	// Define the data structures for weights, inputs, and expected outputs
	weightsData := struct {
		Weights [][][]float64 `json:"weights"`
		Biases  [][]float64   `json:"biases"`
	}{}

	inputData := struct {
		Inputs [][]float64 `json:"inputs"`
	}{}

	expectedData := struct {
		Expected []float64 `json:"outputs"`
	}{}

	// Decode the JSON files into the structs
	_ = json.NewDecoder(weightsFile).Decode(&weightsData)
	_ = json.NewDecoder(inputsFile).Decode(&inputData)
	_ = json.NewDecoder(outputsFile).Decode(&expectedData)

	// Create the circuit and initialize it
	assignment := &ProveModelCircuit{}

	for layer := 0; layer < len(weightsData.Weights); layer++ {

		for neuron := 0; neuron < len(weightsData.Weights[layer]); neuron++ {

			for j := 0; j < len(weightsData.Weights[layer][neuron]); j++ {
				scaledWeight := new(big.Int).SetInt64(int64(weightsData.Weights[layer][neuron][j] * 1000))
				assignment.Weights[layer][neuron][j] = frontend.Variable(scaledWeight)
			}
		}
	}

	// Initialize Biases
	for layer := 0; layer < len(weightsData.Biases); layer++ {
		for i := 0; i < len(weightsData.Biases[layer]); i++ {
			scaledBias := new(big.Int).SetInt64(int64(weightsData.Biases[layer][i] * 1000000))
			assignment.Biases[layer][i] = frontend.Variable(scaledBias)
		}
	}

	// Initialize Inputs
	for i := 0; i < len(inputData.Inputs); i++ {
		for j := 0; j < len(inputData.Inputs[i]); j++ {
			scaledInput := new(big.Int).SetInt64(int64(inputData.Inputs[i][j] * 1000))
			assignment.Inputs[i][j] = frontend.Variable(scaledInput)
		}
	}

	// Initialize Expected Outputs
	for i := 0; i < len(expectedData.Expected); i++ {
		scaledExpected := new(big.Int).SetInt64(int64(expectedData.Expected[i] * 1000))
		assignment.Expected[i] = frontend.Variable(scaledExpected)
	}

	var myCircuit ProveModelCircuit
	fmt.Print(assignment.Inputs)
	// Compile and set up the circuit
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &myCircuit)
	if err != nil {
		fmt.Println("Error compiling circuit:", err)
		return
	}

	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		fmt.Println("Error during setup:", err)
		return
	}

	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		fmt.Println("Error creating witness:", err)
		return
	}

	proof, err := groth16.Prove(cs, pk, witness)
	if err != nil {
		fmt.Println("Error proving:", err)
		return
	}

	publicWitness, err := witness.Public()
	if err != nil {
		fmt.Println("Error getting public witness:", err)
		return
	}

	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		fmt.Println("Verification failed")
	} else {
		fmt.Println("Verification succeeded")
	}
}
