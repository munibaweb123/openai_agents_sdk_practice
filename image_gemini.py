#
# A Python script to generate an image using the Imagen 3 model.
# This script uses the correct API method `generate_images` to avoid the 404 error.
#
# Prerequisites:
# pip install google-generativeai
# pip install Pillow
#

import os
import google.generativeai as genai
from PIL import Image
from io import BytesIO

def generate_image(prompt: str):
    """
    Generates an image using the 'imagen-3.0-generate-002' model.

    Args:
        prompt (str): The text prompt for image generation.
    """
    try:
        # Load the API key from an environment variable.
        # This is a best practice for security.
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        # Initialize the Generative AI client
        genai.configure(api_key=api_key)
        
        # Use the correct API method: `models.generate_images`
        # This is the key change from the previous script.
        print(f"Generating image for prompt: '{prompt}'...")
        
        response = genai.models.generate_images(
            model='imagen-3.0-generate-002',
            prompt=prompt
        )

        # Check if the response contains generated images
        if not response.images:
            print("Error: The model did not return any images.")
            return

        # Process and save the first generated image
        # The API returns a list of images, we'll use the first one.
        generated_image_bytes = response.images[0].image_bytes
        
        # Open the image using the Pillow library
        img = Image.open(BytesIO(generated_image_bytes))

        # Save the image to a file. You can change the filename.
        filename = "futuristic_cityscape.png"
        img.save(filename)
        print(f"Image successfully generated and saved as '{filename}'.")

    except Exception as e:
        print(f"An error occurred during image generation: {e}")

if __name__ == "__main__":
    # Define the image generation prompt
    image_prompt = 'A vibrant oil painting of a futuristic cityscape with flying cars and neon lights, in the style of impressionism.'
    
    # Run the image generation function
    generate_image(image_prompt)
