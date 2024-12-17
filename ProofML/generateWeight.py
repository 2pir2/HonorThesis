import json
import random

def relu(x):
    """ReLU activation function."""
    return max(0, x)

def round_to_3_decimal_places(value):
    """Round to 3 decimal places."""
    return round(value, 3)

def generate_weights_and_biases(layers, neurons_per_layer):
    """Generate weights and biases for the neural network."""
    weights = []
    biases = []
    for i in range(layers):
        # Generate weights for each neuron
        layer_weights = [
            [round_to_3_decimal_places(random.uniform(-1, 1)) for _ in range(neurons_per_layer)] 
            for _ in range(neurons_per_layer)
        ]
        weights.append(layer_weights)

        # Generate biases for each neuron
        layer_biases = [round_to_3_decimal_places(random.uniform(-1, 1)) for _ in range(neurons_per_layer)]
        biases.append(layer_biases)
    
    return weights, biases

def simulate_neural_network(weights, biases, input_vector):
    """Simulate the forward pass of a neural network."""
    current_output = input_vector
    for layer_weights, layer_biases in zip(weights, biases):
        next_output = []
        for neuron_weights, neuron_bias in zip(layer_weights, layer_biases):
            # Weighted sum + bias
            weighted_sum = sum(w * x for w, x in zip(neuron_weights, current_output)) + neuron_bias
            # Apply ReLU activation
            next_output.append(relu(weighted_sum))
        current_output = next_output
    return current_output

def generate_data(layers, neurons_per_layer, num_inputs, output_prefix):
    """Generate weights, biases, inputs, and expected outputs."""
    # Generate weights and biases
    weights, biases = generate_weights_and_biases(layers, neurons_per_layer)
    
    # Generate random inputs
    inputs = [
        [round_to_3_decimal_places(random.uniform(0, 1)) for _ in range(neurons_per_layer)]
        for _ in range(num_inputs)
    ]
    
    # Compute expected outputs
    outputs = []
    for input_vector in inputs:
        final_output = simulate_neural_network(weights, biases, input_vector)
        outputs.append(final_output.index(max(final_output)))  # Argmax of the final layer
    
    # Save weights.json
    with open(f"{output_prefix}_weights.json", "w") as f:
        json.dump({"weights": weights, "biases": biases}, f, indent=4)
    
    # Save inputs.json
    with open(f"{output_prefix}_inputs.json", "w") as f:
        json.dump({"inputs": inputs}, f, indent=4)
    
    # Save outputs.json
    with open(f"{output_prefix}_outputs.json", "w") as f:
        json.dump({"outputs": outputs}, f, indent=4)

# Generate files for different sizes
sizes = [
    (2, 5, 10),    # (layers, neurons per layer, number of inputs)
    (10, 10, 20),  # (layers, neurons per layer, number of inputs)
    (20, 50, 50),  # (layers, neurons per layer, number of inputs)
    (100, 100, 100) # (layers, neurons per layer, number of inputs)
]
for idx, (layers, neurons, num_inputs) in enumerate(sizes):
    output_prefix = f"size_{layers * neurons}"
    generate_data(layers, neurons, num_inputs, output_prefix)
