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

    other_images = []
    for other_image in other_image_filenames:
        other_images.append(Image.open(other_image))

    other_inputs = processor(images=other_images, return_tensors="pt", padding=True)

    with torch.no_grad():
        other_embeddings = model.get_image_features(**other_inputs)

    # calculate cosine similarity to base image for each "other image"
    cos_sim = torch.nn.functional.cosine_similarity(base_embeddings, other_embeddings, dim=1)

    # convert to percentage for better printout
    similarity_percentages = (cos_sim + 1) / 2 * 100

    for i, similarity in enumerate(similarity_percentages):
        print(f"Similarity with {other_image_filenames[i]}: {similarity.item():.2f}%")

###

print("start")

# load CLIP model
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

base_embeddings = load_base_embeddings()
if base_embeddings is None:
    # if the base embeddings don't exist or are invalid, process the base image and save to file
    print("Creating new base embeddings...")
    base_image = Image.open(BASE_IMAGE_PATH)
    base_inputs = processor(images=base_image, return_tensors="pt", padding=True)

    with torch.no_grad():
        base_embeddings = model.get_image_features(**base_inputs)

    torch.save(base_embeddings, BASE_EMBEDDINGS_PATH)
    print("Created new base embeddings.")
else:
    print("Loaded saved base embeddings.")
print(f"Base embeddings shape: {base_embeddings.shape}")

###

compare_to_other_images(["image2.jpg", "image3.jpg"])

print("end")
