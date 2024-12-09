from transformers import AutoTokenizer, VisionEncoderDecoderModel
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Load the Nougat model and tokenizer
model_name = "facebook/nougat-base"  # Replace with your Nougat model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Image preprocessing pipeline
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = Compose([
        Resize((896, 672)),  # Resize to match model's expected input
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Load and preprocess the image
image_path = "KE.jpeg"
pixel_values = preprocess_image(image_path)

# Ensure the device compatibility
device = torch.device("mps" if torch.has_mps else "cpu")
model = model.to(device)
pixel_values = pixel_values.to(device)

# Perform OCR inference with controlled generation length
outputs = model.generate(
    pixel_values, 
    max_new_tokens=512,  # Adjust for document length
    num_beams=4,         # Improves generation quality by exploring alternatives
    no_repeat_ngram_size=3,  # Avoid repetitive outputs
    length_penalty=2.0,  # Encourage longer sequences
    early_stopping=True  # Stop when model is confident
)

# Decode the OCR text
extracted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Extracted Text:")
print(extracted_text)
