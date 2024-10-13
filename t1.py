from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

###

BASE_IMAGE_PATH = "base.jpg"
BASE_EMBEDDINGS_PATH = "base_embeddings.pt"
MODEL_NAME = "openai/clip-vit-large-patch14"

###

def load_base_embeddings():
    try:
        # try to load the base embeddings from file
        return torch.load(BASE_EMBEDDINGS_PATH, weights_only=True)
    except (FileNotFoundError) as e:
        # if file doesn't exist or contains invalid data, return None
        print(f"Failed to load embeddings from file.")
        return None

###

def compare_to_other_images(other_image_filenames):
    print("Loading and processing other images...")

###

print("start")

# load CLIP model
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

base_embeddings = load_base_embeddings()
print("Creating new base embeddings...")

###

compare_to_other_images(["image2.jpg", "image3.jpg"])

print("end")
