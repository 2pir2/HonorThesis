package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	groth16_bn254 "github.com/consensys/gnark/backend/groth16/bn254"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/test"
)

const (
	pubInputFile = "public.json"
	priInputFile = "private.json"
	vkKeyFile    = "vk.g16vk"
	proofFile    = "proof.g16p"
)

type SudokuCircuit struct {
	IncompleteGrid [9][9]frontend.Variable `gnark:"IncompleteSudoku,public"`
	CompleteGrid   [9][9]frontend.Variable `gnark:"CompleteSudoku"`
}

type Sudoku struct {
	Grid [9][9]int `json:"grid"`
}

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

func TestMakeSudoku(t *testing.T) {
	var incompleteSudoku = Sudoku{
		Grid: [9][9]int{
			{5, 3, 0, 0, 7, 0, 0, 0, 0},
			{6, 0, 0, 1, 9, 5, 0, 0, 0},
			{0, 9, 8, 0, 0, 0, 0, 6, 0},
			{8, 0, 0, 0, 6, 0, 0, 0, 3},
			{4, 0, 0, 8, 0, 3, 0, 0, 1},
			{7, 0, 0, 0, 2, 0, 0, 0, 6},
			{0, 6, 0, 0, 0, 0, 2, 8, 0},
			{0, 0, 0, 4, 1, 9, 0, 0, 5},
			{0, 0, 0, 0, 8, 0, 0, 7, 9},
		},
	}

	var completeSudoku = Sudoku{
		Grid: [9][9]int{
			{5, 3, 4, 6, 7, 8, 9, 1, 2},
			{6, 7, 2, 1, 9, 5, 3, 4, 8},
			{1, 9, 8, 3, 4, 2, 5, 6, 7},
			{8, 5, 9, 7, 6, 1, 4, 2, 3},
			{4, 2, 6, 8, 5, 3, 7, 9, 1},
			{7, 1, 3, 9, 2, 4, 8, 5, 6},
			{9, 6, 1, 5, 3, 7, 2, 8, 4},
			{2, 8, 7, 4, 1, 9, 6, 3, 5},
			{3, 4, 5, 2, 8, 6, 1, 7, 9},
		},
	}
	bb, err := json.Marshal(incompleteSudoku)
	if err != nil {
		t.Fatal(err)
	}
	err = os.WriteFile(pubInputFile, bb, 0644)
	if err != nil {
		t.Fatal(err)
	}
	bb, err = json.Marshal(completeSudoku)
	if err != nil {
		t.Fatal(err)
	}
	err = os.WriteFile(priInputFile, bb, 0644)
	if err != nil {
		t.Fatal(err)
	}
}

func TestCreateProof(t *testing.T) {
	assert := test.NewAssert(t)
	incompleteFile, err := os.ReadFile(pubInputFile)
	assert.NoError(err)
	completeFile, err := os.ReadFile(priInputFile)
	assert.NoError(err)

	var incompleteSudoku Sudoku
	err = json.Unmarshal(incompleteFile, &incompleteSudoku)
	assert.NoError(err)

	var completeSudoku Sudoku
	err = json.Unmarshal(completeFile, &completeSudoku)
	assert.NoError(err)

	// Create the circuit assignment
	assignment := &SudokuCircuit{}
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			assignment.IncompleteGrid[i][j] = frontend.Variable(incompleteSudoku.Grid[i][j])
			assignment.CompleteGrid[i][j] = frontend.Variable(completeSudoku.Grid[i][j])
		}
	}

	var myCircuit SudokuCircuit
	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	assert.NoError(err)
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &myCircuit)
	assert.NoError(err)

	pk, vk, err := groth16.Setup(cs)
	assert.NoError(err)

	proof, err := groth16.Prove(cs, pk, witness)
	assert.NoError(err)

	var (
		vkF    *os.File
		proofF *os.File
	)
	vkF, err = os.Create(vkKeyFile)
	assert.NoError(err)
	defer vkF.Close()
	proofF, err = os.Create(proofFile)
	assert.NoError(err)
	defer proofF.Close()

	_, err = vk.WriteTo(vkF)
	assert.NoError(err)
	_, err = proof.WriteTo(proofF)
	assert.NoError(err)
}

func TestReadProof(t *testing.T) {
	assert := test.NewAssert(t)
	var (
		err    error
		vkF    *os.File
		proofF *os.File
		vk     groth16_bn254.VerifyingKey
		proof  groth16_bn254.Proof
	)

	vkF, err = os.Open(vkKeyFile)
	assert.NoError(err)
	defer vkF.Close()
	proofF, err = os.Open(proofFile)
	assert.NoError(err)
	defer proofF.Close()

	// unmarshal public part and create witness
	var incompleteSudoku Sudoku
	pubB, err := os.ReadFile(pubInputFile)
	assert.NoError(err)
	err = json.Unmarshal(pubB, &incompleteSudoku)
	assert.NoError(err)
	var sudokuCircuitAssignmentPublic SudokuCircuit
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			sudokuCircuitAssignmentPublic.IncompleteGrid[i][j] = frontend.Variable(incompleteSudoku.Grid[i][j])
		}
	}
	pubWit, err := frontend.NewWitness(&sudokuCircuitAssignmentPublic, ecc.BN254.ScalarField(), frontend.PublicOnly())
	assert.NoError(err)

	_, err = vk.ReadFrom(vkF)
	assert.NoError(err)
	_, err = proof.ReadFrom(proofF)
	assert.NoError(err)

	err = groth16.Verify(&proof, &vk, pubWit)
	assert.NoError(err)
}

func main() {
	incompleteFile, err := ioutil.ReadFile("public.json")
	if err != nil {
		fmt.Print(err)
	}

	var incompleteSudoku Sudoku
	_ = json.Unmarshal(incompleteFile, &incompleteSudoku)

	TestReadProof(&testing.T{})

}
