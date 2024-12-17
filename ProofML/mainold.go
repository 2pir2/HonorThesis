// package main

// import (
// 	"encoding/json"
// 	"fmt"
// 	"io/ioutil"
// 	"math/big"
// 	"os"

// 	"github.com/consensys/gnark-crypto/ecc"
// 	"github.com/consensys/gnark/backend/groth16"
// 	"github.com/consensys/gnark/frontend"
// 	"github.com/consensys/gnark/frontend/cs/r1cs"
// )

// // ProveModelCircuit defines the simple circuit that only asserts that the output matches the expected output
// type ProveModelCircuit struct {
// 	ComputedOutput frontend.Variable ;gnark: ",public" // Computed output is passed as public input
// 	Expected       frontend.Variable gnark: ",private" // Expected output
// }

// // Define only checks if the computed output equals the expected output
// func (circuit *ProveModelCircuit) Define(api frontend.API) error {
// 	// Assert that the computed output matches the expected output
// 	api.AssertIsEqual(circuit.ComputedOutput, circuit.Expected)
// 	return nil
// }

// // roundAndScaleDown rounds the last three digits and scales down by 10^3
// func roundAndScaleDown(value *big.Int) *big.Int {
// 	mod := new(big.Int).Mod(value, big.NewInt(1000))
// 	if mod.Cmp(big.NewInt(500)) >= 0 {
// 		value.Add(value, big.NewInt(1000))
// 	}
// 	value.Div(value, big.NewInt(1000))
// 	return value
// }

// func NeuralNetworkOutput(weights [][][]float64, biases [][]float64, input []float64, maxSize int) int {
// 	layerOutputs := make([][]float64, len(weights))
// 	field := new(big.Int).Exp(big.NewInt(2), big.NewInt(254), nil)
// 	negativeBoundary := new(big.Int).Div(field, big.NewInt(2))

// 	// First layer computation
// 	layerOutputs[0] = make([]float64, maxSize)
// 	for i := 0; i < len(biases[0]); i++ {
// 		layerOutputs[0][i] = biases[0][i]
// 		for j := 0; j < len(input); j++ {
// 			multiplied := new(big.Int).Set(float64ToBigInt(weights[0][i][j] * input[j]))
// 			rounded := roundAndScaleDown(multiplied)
// 			rounded = ReLUBigInt(rounded, negativeBoundary)
// 			layerOutputs[0][i] += float64(rounded.Int64())
// 		}
// 	}

// 	// Subsequent layers computation
// 	for layerIdx := 1; layerIdx < len(weights); layerIdx++ {
// 		layerOutputs[layerIdx] = make([]float64, maxSize)
// 		for i := 0; i < len(biases[layerIdx]); i++ {
// 			layerOutputs[layerIdx][i] = biases[layerIdx][i]
// 			for j := 0; j < len(layerOutputs[layerIdx-1]); j++ {
// 				multiplied := new(big.Int).Set(float64ToBigInt(weights[layerIdx][i][j] * layerOutputs[layerIdx-1][j]))
// 				rounded := roundAndScaleDown(multiplied)
// 				rounded = ReLUBigInt(rounded, negativeBoundary)
// 				layerOutputs[layerIdx][i] += float64(rounded.Int64())
// 			}
// 		}
// 	}

// 	// Print the final layer's output matrix
// 	fmt.Println("Final layer output matrix:")
// 	for i := 0; i < len(layerOutputs[len(weights)-1]); i++ {
// 		fmt.Printf("Neuron %d output: %f\n", i, layerOutputs[len(weights)-1][i])
// 	}

// 	// Find the maximum value (argmax) in the final layer output
// 	maxIdx := 0
// 	for i := 1; i < len(layerOutputs[len(weights)-1]); i++ {
// 		if layerOutputs[len(weights)-1][i] > layerOutputs[len(weights)-1][maxIdx] {
// 			maxIdx = i
// 		}
// 	}

// 	return maxIdx
// }

// // ReLUBigInt applies ReLU to big.Int with two's complement handling
// func ReLUBigInt(value, negativeBoundary *big.Int) *big.Int {
// 	zero := big.NewInt(0)
// 	if value.Cmp(negativeBoundary) >= 0 || value.Cmp(zero) <= 0 {
// 		return big.NewInt(0)
// 	}
// 	return value
// }

// // ReLU for regular float values
// func ReLU(x float64) float64 {
// 	if x < 0 {
// 		return 0
// 	}
// 	return x
// }

// // float64ToBigInt converts a float64 to a big.Int by scaling
// func float64ToBigInt(value float64) *big.Int {
// 	scaledValue := value * 1e3
// 	return big.NewInt(int64(scaledValue))
// }

// // getMaxLayerSize finds the largest matrix dimension across all layers
// func getMaxLayerSize(weights [][][]float64) int {
// 	maxSize := 0
// 	for _, layer := range weights {
// 		for _, neuronWeights := range layer {
// 			if len(neuronWeights) > maxSize {
// 				maxSize = len(neuronWeights)
// 			}
// 		}
// 	}
// 	return maxSize
// }

// // padWeights pads a weight matrix to maxSize x maxSize, setting irrelevant indices to 0
// func padWeights(weights [][][]float64, maxSize int) [][][]float64 {
// 	paddedWeights := make([][][]float64, len(weights))
// 	for layerIdx, layer := range weights {
// 		paddedWeights[layerIdx] = make([][]float64, maxSize)
// 		for i := range paddedWeights[layerIdx] {
// 			paddedWeights[layerIdx][i] = make([]float64, maxSize)
// 		}
// 		for neuronIdx, neuronWeights := range layer {
// 			copy(paddedWeights[layerIdx][neuronIdx], neuronWeights)
// 		}
// 	}
// 	return paddedWeights
// }

// // padInput pads an input vector to maxSize
// func padInput(input []float64, maxSize int) []float64 {
// 	paddedInput := make([]float64, maxSize)
// 	copy(paddedInput, input)
// 	return paddedInput
// }

// // padOutput pads an output vector to maxSize
// func padOutput(output float64, maxSize int) []float64 {
// 	paddedOutput := make([]float64, maxSize)
// 	paddedOutput[0] = output
// 	return paddedOutput
// }

// func main() {
// 	weightsFile, err := os.Open("weights.json")
// 	if err != nil {
// 		fmt.Println("Error opening weights file:", err)
// 		return
// 	}
// 	defer weightsFile.Close()

// 	inputsFile, err := os.Open("inputs.json")
// 	if err != nil {
// 		fmt.Println("Error opening inputs file:", err)
// 		return
// 	}
// 	defer inputsFile.Close()

// 	outputsFile, err := os.Open("outputs.json")
// 	if err != nil {
// 		fmt.Println("Error opening outputs file:", err)
// 		return
// 	}
// 	defer outputsFile.Close()

// 	weightsByteValue, err := ioutil.ReadAll(weightsFile)
// 	if err != nil {
// 		fmt.Println("Error reading weights file:", err)
// 		return
// 	}

// 	inputsByteValue, err := ioutil.ReadAll(inputsFile)
// 	if err != nil {
// 		fmt.Println("Error reading inputs file:", err)
// 		return
// 	}

// 	outputsByteValue, err := ioutil.ReadAll(outputsFile)
// 	if err != nil {
// 		fmt.Println("Error reading outputs file:", err)
// 		return
// 	}

// 	var weightsData struct {
// 		Weights [][][]float64 json:"weights"
// 		Biases  [][]float64   json:"biases"
// 	}

// 	var inputData struct {
// 		Inputs [][]float64 json:"inputs"
// 	}

// 	var expectedData struct {
// 		Expected []float64 json:"outputs"
// 	}

// 	err = json.Unmarshal(weightsByteValue, &weightsData)
// 	if err != nil {
// 		fmt.Println("Error unmarshalling weights JSON:", err)
// 		return
// 	}

// 	err = json.Unmarshal(inputsByteValue, &inputData)
// 	if err != nil {
// 		fmt.Println("Error unmarshalling inputs JSON:", err)
// 		return
// 	}

// 	err = json.Unmarshal(outputsByteValue, &expectedData)
// 	if err != nil {
// 		fmt.Println("Error unmarshalling outputs JSON:", err)
// 		return
// 	}

// 	maxSize := getMaxLayerSize(weightsData.Weights)
// 	paddedWeights := padWeights(weightsData.Weights, maxSize)

// 	for i := 0; i < len(inputData.Inputs); i++ {
// 		paddedInput := padInput(inputData.Inputs[i], maxSize)
// 		paddedOutput := padOutput(expectedData.Expected[i], maxSize)

// 		computedOutput := NeuralNetworkOutput(paddedWeights, weightsData.Biases, paddedInput, maxSize)

// 		assignment := &ProveModelCircuit{
// 			ComputedOutput: frontend.Variable(computedOutput),
// 			Expected:       frontend.Variable(int(paddedOutput[0])),
// 		}

// 		var myCircuit ProveModelCircuit
// 		witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
// 		if err != nil {
// 			fmt.Println("Error creating witness:", err)
// 			return
// 		}

// 		cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &myCircuit)
// 		if err != nil {
// 			fmt.Println("Error compiling circuit:", err)
// 			return
// 		}

// 		pk, vk, err := groth16.Setup(cs)
// 		if err != nil {
// 			fmt.Println("Error during setup:", err)
// 			return
// 		}

// 		proof, err := groth16.Prove(cs, pk, witness)
// 		if err != nil {
// 			fmt.Println("Error Proving: ", err)
// 			return
// 		}

// 		publicWitness, err := witness.Public()
// 		if err != nil {
// 			fmt.Println("Error getting public witness:", err)
// 			return
// 		}

// 		errverify := groth16.Verify(proof, vk, publicWitness)
// 		if errverify != nil {
// 			fmt.Println("Error in Verifying: ", errverify)
// 		} else {
// 			fmt.Printf("Verification succeeded for input %d\n", i+1)
// 		}
// 	}
// }