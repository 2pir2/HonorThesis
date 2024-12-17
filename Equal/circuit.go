package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math/big"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/algebra/field"
	"github.com/consensys/gnark/cs/r1cs"
	"github.com/consensys/gnark/crypto/ecc"
	"github.com/consensys/gnark/crypto/groth16"
)

type RandomNumberCircuit struct {
	Seed      frontend.Variable `gnark:",public"` // Public input: Seed
	Expected  frontend.Variable `gnark:",public"` // Public input: Expected seed
	Generated frontend.Variable `gnark:",public"` // Public output: Generated random number
}

func (circuit *RandomNumberCircuit) Define(api frontend.API) error {
	// Declare constants as frontend.Variable
	a := frontend.Variable(1664525)      // Multiplier
	c := frontend.Variable(1013904223)  // Increment
	m := frontend.Variable(1 << 32)     // Modulus (2^32)

	// Ensure the seed matches the expected value
	api.AssertIsEqual(circuit.Seed, circuit.Expected)

	// LCG computation: (a * seed + c) % m
	temp := api.Add(api.Mul(a, circuit.Seed), c) // temp = a * seed + c

	// Compute division (temp / m) and multiply by m
	quotient := api.Div(temp, m) // quotient = temp / m
	modulusResult := api.Sub(temp, api.Mul(quotient, m)) // remainder = temp - (quotient * m)

	// Ensure the generated number matches the modulus result
	api.AssertIsEqual(circuit.Generated, modulusResult)

	return nil
}

func main() {
	// Step 1: Read the seed from a JSON file
	type SeedData struct {
		Seed int64 `json:"seed"`
	}
	data, err := ioutil.ReadFile("seed.json")
	if err != nil {
		log.Fatalf("Failed to read seed file: %v", err)
	}
	var seedData SeedData
	if err := json.Unmarshal(data, &seedData); err != nil {
		log.Fatalf("Failed to parse seed JSON: %v", err)
	}

	// Input values for the circuit
	seed := big.NewInt(seedData.Seed)          // Read seed from JSON
	expectedSeed := big.NewInt(seedData.Seed)  // Expected seed (same as input for verification)

	// Linear Congruential Generator (LCG) constants
	a := big.NewInt(1664525)      // Multiplier
	c := big.NewInt(1013904223)  // Increment
	m := big.NewInt(1 << 32)     // Modulus (2^32)

	// Compute the expected generated number using the same formula
	generated := new(big.Int).Add(new(big.Int).Mul(seed, a), c)
	generated.Mod(generated, m)

	// Step 2: Define the circuit and the assignment
	var myCircuit RandomNumberCircuit
	assignment := &RandomNumberCircuit{
		Seed:      seed,
		Expected:  expectedSeed,
		Generated: generated,
	}

	// Compile the circuit
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &myCircuit)
	if err != nil {
		log.Fatalf("Failed to compile circuit: %v", err)
	}

	// Create a witness
	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		log.Fatalf("Failed to create witness: %v", err)
	}

	// Step 3: Setup
	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		log.Fatalf("Setup failed: %v", err)
	}

	// Step 4: Prove
	proof, errproof := groth16.Prove(cs, pk, witness)
	fmt.Println("Error Proving: ", errproof)

	// Generate the public witness
	publicWitness, err := witness.Public()
	if err != nil {
		log.Fatalf("Failed to generate public witness: %v", err)
	}

	// Step 5: Verify
	errverify := groth16.Verify(proof, vk, publicWitness)
	fmt.Println("Error Verifying: ", errverify)

	// Final Verification Check
	if errverify == nil && errproof == nil {
		fmt.Println("Verification Succeeded!")
	} else {
		fmt.Println("Verification Failed!")
	}
}
