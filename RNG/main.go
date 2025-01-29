package main

import (
	"errors"
	"fmt"
	"math/big"
)

// Define the hint function for modular arithmetic
func smallModHint(mod *big.Int, inputs []*big.Int, outputs []*big.Int) error {
	// Computes a % r = b
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
	return nil
}

// SmallMod uses the smallModHint to compute the quotient and remainder
func SmallMod(a, r *big.Int) (quo, rem *big.Int) {
	// Initialize output variables
	rem = new(big.Int)
	quo = new(big.Int)

	// Use the hint function to compute modular division
	err := smallModHint(nil, []*big.Int{a, r}, []*big.Int{rem, quo})
	if err != nil {
		panic(fmt.Sprintf("Error in modular computation: %v", err))
	}

	return quo, rem
}

func main() {
	// Linear Congruential Generator (LCG) constants
	a := big.NewInt(1664525)    // Multiplier
	c := big.NewInt(1013904223) // Increment
	m := big.NewInt(100)        // Modulus (2^32)
	seed := big.NewInt(12345)   // Example seed

	// Step 1: Compute the intermediate result temp = a * seed + c
	temp := new(big.Int).Add(new(big.Int).Mul(seed, a), c)

	// Step 2: Use SmallMod to compute the quotient and remainder
	quo, rem := SmallMod(temp, m)

	// Step 3: Print the results
	fmt.Println("Seed:", seed)
	fmt.Println("Temp:", temp)
	fmt.Println("Quotient:", quo)
	fmt.Println("Remainder:", rem)
}
