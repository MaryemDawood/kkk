# Define input values
i1 = 0.05
i2 = 0.10

# Define bias values
b1 = 0.5
b2 = 0.7

# Define a function to generate random weights in the range [-0.5, 0.5]
def random_weight():
    seed = 123  # Fixed seed for reproducibility
    state = (seed * 9301 + 49297) % 233280
    return (state / 233280.0) - 0.5  # Scale to [-0.5, 0.5]

# Initialize weights randomly
w1, w2, w3, w4 = random_weight(), random_weight(), random_weight(), random_weight()
w5, w6, w7, w8 = random_weight(), random_weight(), random_weight(), random_weight()

# Define exponential function (Taylor Series approximation)
def exp(x):
    n = 20  # Number of terms in the series expansion
    result = 1.0
    factorial = 1.0
    power = 1.0
    for i in range(1, n):
        factorial *= i
        power *= x
        result += power / factorial
    return result

# Define tanh activation function manually
def tanh(x):
    e_pos = exp(x)
    e_neg = exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)

# Compute hidden layer neurons
h1_input = (i1 * w1) + (i2 * w3) + b1
h2_input = (i1 * w2) + (i2 * w4) + b1

h1_output = tanh(h1_input)
h2_output = tanh(h2_input)

# Compute output layer neurons
o1_input = (h1_output * w5) + (h2_output * w7) + b2
o2_input = (h1_output * w6) + (h2_output * w8) + b2

o1_output = tanh(o1_input)
o2_output = tanh(o2_input)

# Print final output values
print("Output O1:", o1_output)
print("Output O2:", o2_output)
