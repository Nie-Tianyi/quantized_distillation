import torch


def main():
    print("Hello from quantized-distillation!")
    print("PyTorch version:", torch.version.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)


if __name__ == "__main__":
    main()
