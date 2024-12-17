package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

// SudokuCircuit defines the constraints for a Sudoku puzzle
type SudokuCircuit struct {
	IncompleteGrid [9][9]frontend.Variable `gnark:"IncompleteSudoku,public"`
	CompleteGrid   [9][9]frontend.Variable `gnark:"CompleteSudoku"`
}

// Define the constraints for the SudokuCircuit
func (circuit *SudokuCircuit) Define(api frontend.API) error {
	// Constraint 1: Each cell value in the CompleteGrid must be between 1 and 9
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			api.AssertIsLessOrEqual(circuit.CompleteGrid[i][j], 9)
			api.AssertIsLessOrEqual(1, circuit.CompleteGrid[i][j])
		}
	}

	// Constraint 2: Each row in the CompleteGrid must contain unique values
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			for k := j + 1; k < 9; k++ {
				api.AssertIsDifferent(circuit.CompleteGrid[i][j], circuit.CompleteGrid[i][k])
			}
		}
	}

	// Constraint 3: Each column in the CompleteGrid must contain unique values
	for j := 0; j < 9; j++ {
		for i := 0; i < 9; i++ {
			for k := i + 1; k < 9; k++ {
				api.AssertIsDifferent(circuit.CompleteGrid[i][j], circuit.CompleteGrid[k][j])
			}
		}
	}

	// Constraint 4: Each 3x3 sub-grid in the CompleteGrid must contain unique values
	for boxRow := 0; boxRow < 3; boxRow++ {
		for boxCol := 0; boxCol < 3; boxCol++ {
			for i := 0; i < 9; i++ {
				for j := i + 1; j < 9; j++ {
					row1 := boxRow*3 + i/3
					col1 := boxCol*3 + i%3
					row2 := boxRow*3 + j/3
					col2 := boxCol*3 + j%3
					api.AssertIsDifferent(circuit.CompleteGrid[row1][col1], circuit.CompleteGrid[row2][col2])
				}
			}
		}
	}

	// Constraint 5: The values in the IncompleteGrid must match the CompleteGrid where provided
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			isCellGiven := api.IsZero(circuit.IncompleteGrid[i][j])
			api.AssertIsEqual(api.Select(isCellGiven, circuit.CompleteGrid[i][j], circuit.IncompleteGrid[i][j]), circuit.CompleteGrid[i][j])
		}
	}

	return nil
}

// Proof, VerificationKey, and other necessary structs
type Point struct {
	X string `json:"X"`
	Y string `json:"Y"`
}

type Quadratic struct {
	A0 string `json:"A0"`
	A1 string `json:"A1"`
}

type Bs struct {
	X Quadratic `json:"X"`
	Y Quadratic `json:"Y"`
}

type CommitmentPok struct {
	X int `json:"X"`
	Y int `json:"Y"`
}

type Proof struct {
	Ar           Point         `json:"Ar"`
	Krs          Point         `json:"Krs"`
	Bs           Bs            `json:"Bs"`
	Commitments  []interface{} `json:"Commitments"`
	CommitmentPok CommitmentPok `json:"CommitmentPok"`
}

type G1Key struct {
	Alpha Point   `json:"Alpha"`
	Beta  Point   `json:"Beta"`
	Delta Point   `json:"Delta"`
	K     []Point `json:"K"`
}

type G2Key struct {
	Beta  struct {
		X Quadratic `json:"X"`
		Y Quadratic `json:"Y"`
	} `json:"Beta"`
	Delta struct {
		X Quadratic `json:"X"`
		Y Quadratic `json:"Y"`
	} `json:"Delta"`
	Gamma struct {
		X Quadratic `json:"X"`
		Y Quadratic `json:"Y"`
	} `json:"Gamma"`
}

type CommitmentKey struct {
	G struct {
		X Quadratic `json:"X"`
		Y Quadratic `json:"Y"`
	} `json:"G"`
	GRootSigmaNeg struct {
		X Quadratic `json:"X"`
		Y Quadratic `json:"Y"`
	} `json:"GRootSigmaNeg"`
}

type VerificationKey struct {
	G1                   G1Key          `json:"G1"`
	G2                   G2Key          `json:"G2"`
	CommitmentKey        CommitmentKey  `json:"CommitmentKey"`
	PublicAndCommitmentCommitted []interface{} `json:"PublicAndCommitmentCommitted"`
}

type PublicWitness struct {
	Grid [9][9]int `json:"grid"`
}

func main() {
	// Paths to JSON files
	proofFilePath := "proof.json"
	vkFilePath := "vk.json"
	witnessFilePath := "public_witness.json"

	// Read and unmarshal proof.json
	proofData, err := os.ReadFile(proofFilePath)
	if err != nil {
		fmt.Println("Error reading proof JSON file:", err)
		return
	}

	var proof Proof
	err = json.Unmarshal(proofData, &proof)
	if err != nil {
		fmt.Println("Error unmarshalling proof JSON:", err)
		return
	}

	// Read and unmarshal vk.json
	vkData, err := os.ReadFile(vkFilePath)
	if err != nil {
		fmt.Println("Error reading verification key JSON file:", err)
		return
	}

	var vk VerificationKey
	err = json.Unmarshal(vkData, &vk)
	if err != nil {
		fmt.Println("Error unmarshalling verification key JSON:", err)
		return
	}

	// Read and unmarshal public_witness.json
	witnessData, err := os.ReadFile(witnessFilePath)
	if err != nil {
		fmt.Println("Error reading public witness JSON file:", err)
		return
	}

	var publicWitness PublicWitness
	err = json.Unmarshal(witnessData, &publicWitness)
	if err != nil {
		fmt.Println("Error unmarshalling public witness JSON:", err)
		return
	}

	// Create the SudokuCircuit with the public witness data
	var circuit SudokuCircuit
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			circuit.IncompleteGrid[i][j] = frontend.Variable(publicWitness.Grid[i][j])
		}
	}

	// Compile the circuit (using the public inputs only)
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		fmt.Println("Error compiling circuit:", err)
		return
	}

	// Create the public witness for verification
	witness, err := frontend.NewWitness(&circuit, ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err != nil {
		fmt.Println("Error creating public witness:", err)
		return
	}

	// Verify the proof
	err = groth16.Verify(proof, vk, witness)
	if err != nil {
		fmt.Println("Proof verification failed:", err)
	} else {
		fmt.Println("Proof verified successfully.")
	}
}
