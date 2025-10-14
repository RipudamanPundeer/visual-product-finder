import json
import os
from PIL import Image
from sentence_transformers import SentenceTransformer
import pandas as pd

# --- CONFIGURATION ---
# Adjust these paths if your folder structure is different.
# -------------------------------------------------------------------
CONFIG = {
    # Path to the JSON file containing product metadata.
    # This should point to the file with local image paths.
    "source_json_path": "../public/products_local.json",

    # Folder where the corresponding images are saved.
    "image_folder": "../public",

    # The file where the final embeddings will be saved.
    "output_path": "../public/product_embeddings.json",

    # The key in your JSON that contains the local image path (e.g., '/product-images/shoe.jpg').
    "image_path_key": "url",

    # The key for the unique product ID.
    "id_key": "id",

    # The name of the pre-trained AI model to use.
    "model_name": 'clip-ViT-B-32'
}


# -------------------------------------------------------------------

def generate_embeddings():
    """
    Processes a dataset of images to generate and save their vector embeddings.
    """
    print(f"Loading AI model '{CONFIG['model_name']}'. This may take a moment...")
    # 1. Load the pre-trained Sentence Transformer model
    model = SentenceTransformer(CONFIG['model_name'])

    # 2. Load the product data using pandas for convenience
    try:
        products_df = pd.read_json(CONFIG['source_json_path'])
    except FileNotFoundError:
        print(f"ERROR: Source JSON not found at '{CONFIG['source_json_path']}'")
        return

    total_products = len(products_df)
    print(f"Found {total_products} products. Starting embedding generation...")

    embeddings_data = []

    # 3. Loop through each product in the dataframe
    for index, row in products_df.iterrows():
        image_path_relative = row[CONFIG['image_path_key']]
        product_id = row[CONFIG['id_key']]

        # Construct the full local path to the image
        # The relative path starts with '/', so we remove it to join correctly
        image_path_full = os.path.join(CONFIG['image_folder'], image_path_relative.lstrip('/\\'))

        if not os.path.exists(image_path_full):
            print(f"❗️ SKIPPING: Image not found for product ID {product_id} at '{image_path_full}'")
            continue

        try:
            print(f"Processing ({index + 1}/{total_products}): Product ID {product_id}")

            # Open the image using Pillow
            image = Image.open(image_path_full)

            # Generate the embedding vector for the image
            embedding = model.encode(image)

            # Store the ID and its embedding (converted to a standard list)
            embeddings_data.append({
                "id": product_id,
                "embedding": embedding.tolist()
            })

        except Exception as e:
            print(f"❗️ FAILED to process image for product ID {product_id}. Reason: {e}")

    # 4. Save the collected embeddings to a new JSON file
    with open(CONFIG['output_path'], 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2)

    print(f"\n✅ Embedding generation complete. Data saved to '{CONFIG['output_path']}'.")


if __name__ == "__main__":
    generate_embeddings()