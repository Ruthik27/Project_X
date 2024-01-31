import os
import safetensors.torch as st
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize

# Enable detailed logging for debugging
os.environ['PYTORCH_SHOW_CPP_STACK_TRACES'] = '1'
torch.autograd.set_detect_anomaly(True)

# Set CUDA_VISIBLE_DEVICES to use GPUs 1, 2, and 3
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

# Load the tokenizer
tokenizer_path = './model_hurricane3000/'  # Directory containing tokenizer files
tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

# Load the fine-tuned embeddings using safetensors
embeddings_path = './model_hurricane3000/learned_embeds-steps-3000.safetensors'  # Path to the fine-tuned embeddings
with open(embeddings_path, 'rb') as f:
    embeddings_data = f.read()
learned_embeds_dict = st.load(embeddings_data)
learned_embeds = learned_embeds_dict['<disaster>']  # Replace with the correct key

# Initialize the Stable Diffusion pipeline with the base model
model_id = "CompVis/stable-diffusion-v1-4"  # Replace with a standard model for testing
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Update the text encoder with the fine-tuned embeddings
pipe.text_encoder.get_input_embeddings().weight.data = learned_embeds

# Move the model to GPU
pipe = pipe.to("cuda")

# Preprocess the image
pre_disaster_image_path = "./finaldata/test/pre/hurricane-florence_00000003_pre_disaster_left.png"
pre_disaster_image = Image.open(pre_disaster_image_path)

transform = Compose([
    Resize((512, 512)),  # Adjust the size according to your model's input requirements
    ToTensor()
])
pre_disaster_image = transform(pre_disaster_image)

# Ensure the image tensor has the correct dimensions
if len(pre_disaster_image.shape) == 3:
    pre_disaster_image = pre_disaster_image.unsqueeze(0)
pre_disaster_image = pre_disaster_image.to("cuda")

print("Pre-disaster image shape:", pre_disaster_image.shape)  # Check tensor size

prompt = "Post Disaster Satellite Image of a Hurricane effected area."

# Generate the post-disaster image
try:
    generated_image = pipe(prompt=prompt, init_image=pre_disaster_image, num_inference_steps=50, guidance_scale=7.5).images[0]
except RuntimeError as e:
    print("Runtime error:", e)
    print("Checking CUDA error...")
    torch.cuda.synchronize()  # This will help to get the detailed CUDA error message

    # Handle further debugging or error logging here

# Convert the generated image back to PIL for saving
generated_image = generated_image.cpu().permute(1, 2, 0).numpy()
generated_image = (generated_image * 255).astype('uint8')
generated_image = Image.fromarray(generated_image)

# Save the generated image
output_path = "generated_post_disaster_image.png"
print(f"Saving generated image to {output_path}")
generated_image.save(output_path)
print("Image saved.")
