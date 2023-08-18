# import torch
# import torch.nn as nn
# import argparse
# __all__ = [
#     "efficientnet_b0",
#     "inception-v1",
#     "mobilenet-v2",
#     "resNeXt50",
#     "vgg19",
#     "resnet18",
#     "resnet50",
# ]

# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
#     parser.add_argument(
#         "--model", default="resnet18", type=str, help="model to train the dataset"
#     )
#     args = parser.parse_args()
# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from utils.get_model import get_model

model = get_model("resnet50")

