package main

import (
	"fmt"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

type exampleC struct {
	X [2]frontend.Variable `gnark:",secret"`
	Y frontend.Variable    `gnark:",public"`
}

func (circuit *exampleC) Define(api frontend.API) error {
	api.AssertIsEqual(circuit.X[0], circuit.Y)
	api.AssertIsEqual(circuit.X[1], circuit.Y)
	return nil
}

func main() {
	assignment := &exampleC{}
	assignment.X[0] = 3
	assignment.X[1] = 3
	assignment.Y = frontend.Variable(3)
	var myCircuit exampleC

	cs, _ := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &myCircuit)
	fmt.Print(myCircuit)
	pk, vk, _ := groth16.Setup(cs)
	witness, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	proof, _ := groth16.Prove(cs, pk, witness)
	publicWitness, _ := witness.Public()

	_ = groth16.Verify(proof, vk, publicWitness)

}
