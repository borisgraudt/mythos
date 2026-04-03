import torch
import platform

# versions
print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")
print(f"Mac model: {platform.machine()}")

# test MPS
if torch.backends.mps.is_available():
    print("MPS is available!")
    if torch.backends.mps.is_built():
        print("MPS is build!")
    device = torch.device("mps")
else:
    print("MPS isn't available.")
    device = torch.device("cpu")

print(f"Using device: {device}")

# test tensors
try:
    x = torch.ones(1000, 1000, device=device)
    y = x @ x.T
    print(f"Tensors work on {device}. Shape: {y.shape}")

    # basic model
    model = torch.nn.Linear(512, 512).to(device)
    input_tensor = torch.randn(32, 512, device=device)
    output = model(input_tensor)
    print(f"Model pass on {device}.")

except Exception as e:
    print(f"Error on {device}: {e}")
