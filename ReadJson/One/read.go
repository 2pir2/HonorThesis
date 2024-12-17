package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// Define the structure of the JSON data
type Address struct {
	Street string `json:"street"`
	City   string `json:"city"`
	State  string `json:"state"`
	Zip    string `json:"zip"`
}

type PhoneNumber struct {
	Type   string `json:"type"`
	Number string `json:"number"`
}

type Purchase struct {
	Item  string  `json:"item"`
	Price float64 `json:"price"`
	Date  string  `json:"date"`
}

type Person struct {
	Name            string        `json:"name"`
	Age             int           `json:"age"`
	Email           string        `json:"email"`
	Address         Address       `json:"address"`
	PhoneNumbers    []PhoneNumber `json:"phoneNumbers"`
	Hobbies         []string      `json:"hobbies"`
	IsSubscribed    bool          `json:"isSubscribed"`
	PurchaseHistory []Purchase    `json:"purchaseHistory"`
}

func main() {
	// Provide the path to the JSON file
	filePath := "random.json"

	// Read the JSON file
	data, err := os.ReadFile(filePath)
	if err != nil {
		fmt.Println("Error reading JSON file:", err)
		return
	}
	
	// Print the raw JSON data (optional, for debugging purposes)
	//fmt.Println("Raw JSON data:", string(data))

	// Unmarshal the JSON data into the Go struct
	var person Person
	err = json.Unmarshal(data, &person)
	if err != nil {
		fmt.Println("Error unmarshalling JSON:", err)
		return
	}

	// Print the parsed data
	fmt.Println("Parsed JSON data:", person)
}
