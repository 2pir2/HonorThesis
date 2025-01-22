import json
import sys
from io import StringIO

def relu(x):
    """ReLU activation function."""
    return max(0, x)

def scale_and_convert(value, scale=1000):
    """Scale a float by a given scale factor and convert to an integer."""
    return int(value * scale)

def round_and_scale_down(value, scale=1000):
    """
    Round the last digits according to the scale and scale down.
    - If last digits >= half of scale, round up.
    - Otherwise, keep as is.
    """
    mod = value % scale
    if mod >= scale // 2:
        value += scale
    value -= mod  # Remove fractional digits
    return value // scale

def relu_bigint(value, large_boundary):
    """Apply ReLU to a big integer using a large boundary."""
    if value >= large_boundary or value <= 0:
        return 0
    return value

def neural_network_scaled(weights, biases, input_vector, max_size, scale=1000):
    """Simulate the neural network forward pass with scaling."""
    layer_outputs_scaled = [None] * len(weights)
    layer_outputs_exact = [None] * len(weights)

    # Large boundary for ReLU (independent of scale)
    large_boundary = 1000000000

    # Scale inputs, weights, and biases
    scaled_input_vector = [scale_and_convert(x, scale) for x in input_vector]
    scaled_weights = [[[scale_and_convert(w, scale) for w in neuron_weights] for neuron_weights in layer] for layer in weights]
    scaled_biases = [[scale_and_convert(b, scale*scale) for b in layer_biases] for layer_biases in biases]

    # First layer computation
    layer_outputs_scaled[0] = [0] * max_size
    layer_outputs_exact[0] = [0] * max_size
    for i in range(len(scaled_biases[0])):
        # Exact computation
        layer_outputs_exact[0][i] = biases[0][i]
        for j in range(len(input_vector)):
            tmpAccu = weights[0][i][j] * input_vector[j]
            layer_outputs_exact[0][i] += weights[0][i][j] * input_vector[j]
        layer_outputs_exact[0][i] = relu(layer_outputs_exact[0][i])  # ReLU

        # Scaled computation
        sum_accumulated = scaled_biases[0][i]
        for j in range(len(scaled_input_vector)):
            tmpRound = scaled_weights[0][i][j] * scaled_input_vector[j]
            sum_accumulated += scaled_weights[0][i][j] * scaled_input_vector[j]
        rounded = round_and_scale_down(sum_accumulated, scale)
        layer_outputs_scaled[0][i] = relu_bigint(rounded, large_boundary)

    print(f"\nLayer 1 Add Comparison:")

    # Debug layer 1 comparison
    print(f"\nLayer 1 Outputs Comparison:")
    for i in range(len(layer_outputs_scaled[0])):
        print(f"Neuron {i}: Scaled = {layer_outputs_scaled[0][i]:>8}, Exact = {layer_outputs_exact[0][i]:>12.6f}")

    # Subsequent layers computation
    for layer_idx in range(1, len(weights)):
        layer_outputs_scaled[layer_idx] = [0] * max_size
        layer_outputs_exact[layer_idx] = [0] * max_size
        for i in range(len(scaled_biases[layer_idx])):
            # Exact computation
            layer_outputs_exact[layer_idx][i] = biases[layer_idx][i]
            for j in range(len(layer_outputs_exact[layer_idx - 1])):
                layer_outputs_exact[layer_idx][i] += weights[layer_idx][i][j] * layer_outputs_exact[layer_idx - 1][j]
            layer_outputs_exact[layer_idx][i] = relu(layer_outputs_exact[layer_idx][i])  # ReLU

            # Scaled computation
            sum_accumulated = scaled_biases[layer_idx][i]
            for j in range(len(layer_outputs_scaled[layer_idx - 1])):
                sum_accumulated += scaled_weights[layer_idx][i][j] * layer_outputs_scaled[layer_idx - 1][j]
            rounded = round_and_scale_down(sum_accumulated, scale)
            layer_outputs_scaled[layer_idx][i] = relu_bigint(rounded, large_boundary)

        # Debug subsequent layer comparisons
        print(f"\nLayer {layer_idx + 1} Outputs Comparison:")
        for i in range(len(layer_outputs_scaled[layer_idx])):
            print(f"Neuron {i}: Scaled = {layer_outputs_scaled[layer_idx][i]:>8}, Exact = {layer_outputs_exact[layer_idx][i]:>12.6f}")

    # Final layer's outputs
    print("\nFinal layer output matrix (scaled values):")
    for i, output in enumerate(layer_outputs_scaled[len(weights) - 1]):
        print(f"Neuron {i} output: {output}")

    # Find the maximum value (argmax) in the final layer output
    max_idx = layer_outputs_scaled[len(weights) - 1].index(max(layer_outputs_scaled[len(weights) - 1]))

    return max_idx

# Capture printed output
output_buffer = StringIO()
original_stdout = sys.stdout
sys.stdout = output_buffer

# Load JSON files
with open(r"/Users/hanxu/Desktop/AndyHonorThesis/ProofML/weights.json", "r") as weights_file:
    weights_data = json.load(weights_file)

weights = weights_data["weights"]
biases = weights_data["biases"]

with open(r"/Users/hanxu/Desktop/AndyHonorThesis/ProofML/inputs.json", "r") as inputs_file:
    inputs_data = json.load(inputs_file)

input_vectors = inputs_data["inputs"]

with open(r"/Users/hanxu/Desktop/AndyHonorThesis/ProofML/outputs.json", "r") as outputs_file:
    outputs_data = json.load(outputs_file)

expected_outputs = outputs_data["outputs"]

# Simulate and validate
for input_index, input_vector in enumerate(input_vectors):
    print(f"\nSimulating for Input {input_index + 1}: {input_vector}")
    predicted_output = neural_network_scaled(weights, biases, input_vector, len(weights[0][0]))

    # Compare predicted and expected outputs
    expected_output = expected_outputs[input_index]
    print(f"\nPredicted Output: {predicted_output}, Expected Output: {expected_output}")
    if predicted_output == expected_output:
        print("✅ Match")
    else:
        print("❌ Mismatch")

# Save output to a JSON file
sys.stdout = original_stdout
output_text = output_buffer.getvalue()

# Write output manually line by line
output_lines = output_text.split("\n")
formatted_output = {"output": output_lines}

with open(r"/Users/hanxu/Desktop/AndyHonorThesis/ProofML/simulation_output.json", "w") as json_file:
    json.dump(formatted_output, json_file, indent=4)

print("Simulation results saved to simulation_output.json")
