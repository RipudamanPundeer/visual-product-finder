import os
import json
import requests
import re
import io
from PIL import Image

# --- CONFIGURATION ---
# All settings are defined here for easy modification.
# -------------------------------------------------------------------
CONFIG = {
    # Path to the source JSON file with web URLs.
    "source_json_path": "../public/products.json",

    # Path where the new JSON file with local image paths will be saved.
    "output_json_path": "../public/products_local.json",

    # Folder where the downloaded and converted images will be stored.
    "output_image_folder": "../public/product-images",

    # --- JSON Key Configuration ---
    # Key in the source JSON that holds the image web URL.
    "source_url_key": "url",

    # Key that will be created in the output JSON to hold the new local path.
    "output_path_key": "url",  # Changed to 'url' as requested

    # Key for the product's name (used for naming files).
    "name_key": "name",

    # Key for the product's unique ID (used for naming files).
    "id_key": "id",

    # --- Image Processing ---
    # The format to save all processed images in.
    "output_format": "JPEG",

    # The file extension for the saved images.
    "output_extension": ".jpg",

    # The quality of the saved JPEG images (1-95).
    "image_quality": 70
}


# -------------------------------------------------------------------

def sanitize_filename(name):
    """Takes a string and returns a version that is safe for filenames."""
    name = name.replace(' ', '_')
    name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    return name[:50].lower()


def download_and_process_dataset():
    """
    Reads a JSON file, downloads images, converts them to a standard format,
    and creates a new JSON file with updated local paths.
    """
    print("Starting dataset download and processing...")

    # Create the output directory if it doesn't exist
    if not os.path.exists(CONFIG["output_image_folder"]):
        os.makedirs(CONFIG["output_image_folder"])
        print(f"Created directory: {CONFIG['output_image_folder']}")

    # Load the source JSON file
    try:
        with open(CONFIG["source_json_path"], 'r', encoding='utf-8') as f:
            products = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Source file not found at '{CONFIG['source_json_path']}'")
        return

    new_products_data = []
    total_products = len(products)
    print(f"Found {total_products} products to process.")

    for i, product in enumerate(products):
        # Read data using keys from the CONFIG dictionary
        image_url = product.get(CONFIG["source_url_key"])
        product_id = product.get(CONFIG["id_key"], i + 1)
        product_name = product.get(CONFIG["name_key"])

        if not image_url or not product_name:
            print(f"Skipping product ID {product_id} due to missing URL or name.")
            continue

        try:
            # Create the final filename
            sanitized_name = sanitize_filename(product_name)
            local_filename = f"{sanitized_name}_{product_id}{CONFIG['output_extension']}"
            output_path = os.path.join(CONFIG["output_image_folder"], local_filename)

            print(f"({i + 1}/{total_products}) Processing '{product_name}' -> {local_filename}")

            # Download the image data
            response = requests.get(image_url, stream=True, timeout=15)
            response.raise_for_status()

            # Open the image from downloaded content
            image = Image.open(io.BytesIO(response.content))

            # Convert to RGB if necessary (handles transparent images)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")

            # Save the image in the specified format and quality
            image.save(output_path, CONFIG["output_format"], quality=CONFIG["image_quality"])

            # Create the new product entry for the output JSON
            new_product = product.copy()
            # Remove the old web URL and add the new local path
            del new_product[CONFIG["source_url_key"]]
            new_product[CONFIG["output_path_key"]] = f"/product-images/{local_filename}"
            new_products_data.append(new_product)

        except Exception as e:
            print(f"❗️ Failed to process image for product '{product_name}': {e}")

    # Write the new JSON data to the output file
    with open(CONFIG["output_json_path"], 'w', encoding='utf-8') as f:
        json.dump(new_products_data, f, indent=2)

    print(f"\n✅ Done! All images converted and saved to '{CONFIG['output_image_folder']}'.")
    print(f"✅ New data file created: '{CONFIG['output_json_path']}'")


if __name__ == "__main__":
    download_and_process_dataset()