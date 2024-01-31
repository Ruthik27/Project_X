import rasterio
from rasterio.enums import Resampling
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

def split_tiff_no_overlap(tif_path, output_folder, chunk_size=(512, 512)):
    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height

        # Calculate total chunks for progress bar
        total_chunks = (width // chunk_size[0]) * (height // chunk_size[1])
        print(f"Total chunks to be created: {total_chunks}")

        # Initialize progress bar
        pbar = tqdm(total=total_chunks, desc="Processing Chunks", position=0, leave=True)

        for i in range(0, width, chunk_size[0]):
            for j in range(0, height, chunk_size[1]):
                # Define the window without overlap
                window = rasterio.windows.Window(i, j, chunk_size[0], chunk_size[1])
                chunk_data = src.read(window=window)

                # If the window goes beyond the image dimensions
                if chunk_data.shape[1] != chunk_size[0] or chunk_data.shape[2] != chunk_size[1]:
                    # Create a new array and fill it with zeros (or another value to indicate no data)
                    new_chunk_data = np.zeros((chunk_data.shape[0], chunk_size[0], chunk_size[1]), dtype=chunk_data.dtype)
                    new_chunk_data[:, :chunk_data.shape[1], :chunk_data.shape[2]] = chunk_data
                    chunk_data = new_chunk_data

                # Convert chunk data to an image and save as PNG
                chunk_image = Image.fromarray(np.transpose(chunk_data, (1, 2, 0)))
                output_path = os.path.join(output_folder, f"chunk_{i}_{j}.png")
                chunk_image.save(output_path)
                # Update progress bar
                pbar.update(1)

        # Close progress bar
        pbar.close()

# Define the input TIFF file and output folder path
tif_path = '/mnt/my_network_drive/64fef648649d390001c47539.tif'  # The mounted path to the TIFF file on the server
output_folder = '/mnt/my_network_drive/chunks_folder'  # The mounted path to the output folder on the server

# Create the output directory if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Run the function with the correct paths
split_tiff_no_overlap(tif_path, output_folder)
