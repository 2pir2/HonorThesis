package main

import (
	"errors"
	"fmt"
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint/solver"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/test"
)

func smallModHint(mod *big.Int, inputs []*big.Int, outputs []*big.Int) error {
	// computes a % r = b
	// inputs[0] = a -- input
	// inputs[1] = r -- modulus
	// outputs[0] = b -- remainder
	// outputs[1] = (a-b)/r -- quotient
	if len(outputs) != 2 {
		return errors.New("expected 2 outputs")
	}
	if len(inputs) != 2 {
		return errors.New("expected 2 inputs")
	}
	outputs[1].QuoRem(inputs[0], inputs[1], outputs[0])
	fmt.Println(outputs[0], outputs[1], inputs[0], inputs[1])
	return nil
}

func SmallMod(api frontend.API, a, r frontend.Variable) (quo, rem frontend.Variable) {

	res, err := api.Compiler().NewHint(smallModHint, 2, a, r)
	if err != nil {
		panic(err)
	}
	rem = res[0]
	quo = res[1]

	// to prevent against overflows, we assume that the inputs are small relative to the native field
	nbBits := api.Compiler().Field().BitLen()/2 - 2
	bound := new(big.Int).Lsh(big.NewInt(1), uint(nbBits))

	fmt.Println("Original Number:", 4)
	fmt.Println("Field ELement Representation:", frontend.Variable(4))

	api.AssertIsLessOrEqual(rem, bound)
	api.AssertIsLessOrEqual(quo, bound)

	api.AssertIsEqual(a, api.Add(api.Mul(quo, r), rem))
	return quo, rem
}

type Circuit struct {
	A frontend.Variable `gnark:",public"`
	R frontend.Variable `gnark:",private"`
}

func (c *Circuit) Define(api frontend.API) error {
	c.A = 232213
	c.R = 12
	quo, rem := SmallMod(api, c.A, 1000)
	api.Println("Original Number", c.A)
	api.Println("Dividor", c.R)
	api.Println("Quotient", quo)
	api.Println("Remainder", rem)
	return nil
}

func TestSmallMod(t *testing.T) {
	assert := test.NewAssert(t)

	// hint needs to be provided manually to the solver. Otherwise it doesn't know how to compute the value
	solver.RegisterHint(smallModHint)

	assert.CheckCircuit(&Circuit{}, test.WithValidAssignment(&Circuit{A: 1234000, R: 1000}))

}

func main() {
	solver.RegisterHint(smallModHint)
	assignment := &Circuit{}
	assignment.A = 121
	assignment.R = 10
	fmt.Print(assignment)
	var myCircuit Circuit
	cs, _ := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &myCircuit)
	pk, vk, _ := groth16.Setup(cs)
	witness, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	proof, _ := groth16.Prove(cs, pk, witness)
	publicWitness, _ := witness.Public()
	_ = groth16.Verify(proof, vk, publicWitness)
	fmt.Print("Verification Success")
	// a := frontend.Variable(121)
	// b := frontend.Variable(10)

	// quo, rem := SmallMod(nil, a, b)
	// fmt.Print(quo, " ", rem)
}

// func main() {
// 	// Register the hint function
// 	solver.RegisterHint(smallModHint)

// 	quo, rem := SmallMod(frontend.API, 10, 121)

// 	// Print the results
// 	fmt.Printf("Quotient: %v, Remainder: %v\n", quo, rem)
// }
