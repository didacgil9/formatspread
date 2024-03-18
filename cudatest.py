import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the CUDA device count
    cuda_device_count = torch.cuda.device_count()
    print(f"Found {cuda_device_count} CUDA device(s)")

    # Get information about each CUDA device
    for i in range(cuda_device_count):
        device = torch.cuda.get_device_properties(i)
        compute_capability = torch.cuda.get_device_capability(i)
        print(f"Device {i}: {device.name}, Compute Capability: {compute_capability}")
else:
    print("CUDA is not available. Make sure your GPU drivers and CUDA Toolkit are properly installed.")
