package main

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
)

// Struct to represent the data in initialPoint.json
type InitialData struct {
	InitialPoint [4]float64 `json:"initialPoint"`
	Boundry      float64    `json:"boundry"`
}

// Function to calculate Euclidean distance between two 4D points
func euclideanDistance(p1, p2 [4]float64) float64 {
	sum := 0.0
	for i := 0; i < 4; i++ {
		sum += (p1[i] - p2[i]) * (p1[i] - p2[i])
	}
	return math.Sqrt(sum)
}

// Function to generate cryptographically secure random floating-point numbers
func secureRandomFloat64(min, max float64) (float64, error) {
	// Create a 64-bit random number
	var b [8]byte
	_, err := rand.Read(b[:])
	if err != nil {
		return 0, err
	}

	// Convert bytes into a float64 value
	randomInt := binary.LittleEndian.Uint64(b[:])
	randomFloat := float64(randomInt) / (1 << 64) // Normalize to [0, 1)

	// Scale and shift the random float to the desired range [min, max]
	return min + randomFloat*(max-min), nil
}

// Function to generate a random point within a max distance from the reference point
func generatePointWithinDistance(referencePoint [4]float64, maxDistance float64) ([4]float64, error) {
	var newPoint [4]float64
	for {
		// Generate random values by adding small perturbations to each component of the reference point
		for i := 0; i < 4; i++ {
			perturbation, err := secureRandomFloat64(-maxDistance, maxDistance)
			if err != nil {
				return [4]float64{}, err
			}
			newPoint[i] = referencePoint[i] + perturbation
		}

		// Calculate the Euclidean distance between the new point and the reference point
		distance := euclideanDistance(newPoint, referencePoint)

		// If the distance is within the limit, round each component to 2 decimal places and return the point
		if distance <= maxDistance {
			for i := 0; i < 4; i++ {
				newPoint[i] = math.Round(newPoint[i]*100) / 100 // Round to 2 decimal places
			}
			break
		}
	}
	return newPoint, nil
}

func main() {
	// Step 1: Read the initialPoint.json file
	fileData, err := ioutil.ReadFile("initialPoint.json")
	if err != nil {
		fmt.Println("Error reading file:", err)
		return
	}

	// Step 2: Unmarshal the JSON data into the InitialData struct
	var initialData InitialData
	err = json.Unmarshal(fileData, &initialData)
	if err != nil {
		fmt.Println("Error unmarshalling JSON:", err)
		return
	}

	// Step 3: Use the initial point and boundary from the JSON file
	referencePoint := initialData.InitialPoint
	maxDistance := initialData.Boundry

	// Step 4: Generate 50 random points based on the data from the file
	numPoints := 50
	generatedPoints := make([][4]float64, numPoints)
	for i := 0; i < numPoints; i++ {
		point, err := generatePointWithinDistance(referencePoint, maxDistance)
		if err != nil {
			fmt.Println("Error generating point:", err)
			return
		}
		generatedPoints[i] = point
	}

	// Step 5: Prepare the data in the required JSON format
	data := make(map[string][][]float64)
	data["inputs"] = make([][]float64, numPoints)
	for i, point := range generatedPoints {
		data["inputs"][i] = point[:]
	}

	// Step 6: Write the generated points to inputs.json
	file, err := os.Create("inputs.json")
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ") // Format with indentation
	err = encoder.Encode(data)
	if err != nil {
		fmt.Println("Error writing to file:", err)
		return
	}

	err = file.Close() // Ensure file is closed properly
	if err != nil {
		fmt.Println("Error closing file:", err)
		return
	}

	fmt.Println("Generated 50 random points and saved them to inputs.json")
}
