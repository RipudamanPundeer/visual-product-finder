import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import io
import requests
import base64

# --- INITIAL APP CONFIGURATION ---
st.set_page_config(
    page_title="Visual Product Matcher",
    page_icon="🤖",
    layout="wide"
)


# --- HELPER FUNCTIONS ---
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)


def image_to_base64(image_path, size=(300, 300)):
    """
    Resizes an image to fit within a given size while maintaining aspect ratio,
    adds padding to make it square, and converts to a base64 string.
    """
    img = Image.open(image_path)

    # Create a thumbnail of the image, which preserves aspect ratio
    img.thumbnail(size, Image.Resampling.LANCZOS)

    # Create a new image with a white background
    new_img = Image.new("RGB", size, (255, 255, 255))

    # Paste the thumbnail onto the center of the new image
    paste_position = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
    new_img.paste(img, paste_position)

    # Convert to base64
    buffered = io.BytesIO()
    new_img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def inject_custom_css():
    st.markdown("""
        <style>
            .product-card {
                border: 1px solid #e6e6e6;
                border-radius: 10px;
                padding: 10px;
                text-align: center;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                margin-bottom: 20px;
            }
            .product-image-container {
                width: 100%;
                padding-top: 100%; /* 1:1 Aspect Ratio */
                position: relative;
                overflow: hidden;
                border-radius: 5px;
            }
            .product-image-container img {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
            .product-details {
                margin-top: 10px;
            }
        </style>
    """, unsafe_allow_html=True)


# --- LOAD MODELS AND DATA ---
@st.cache_resource
def load_model():
    return SentenceTransformer('clip-ViT-B-32')


@st.cache_data
def load_data():
    products_df = pd.read_json('public/products_local.json')
    embeddings_df = pd.read_json('public/product_embeddings.json')
    embeddings_dict = {item['id']: item['embedding'] for item in embeddings_df.to_dict('records')}
    products_df['embedding'] = products_df['id'].map(embeddings_dict)
    return products_df


# --- MAIN APP ---
inject_custom_css()
model = load_model()
products_df = load_data()

if 'results' not in st.session_state:
    st.session_state.results = None

st.title("Visual Product Matcher 🛍️")
st.write("Choose your search method, provide an image, and find visually similar products.")

st.sidebar.title("Search an Image")
input_method = st.sidebar.selectbox(
    "Choose your input method", ["Upload an Image", "Paste Image URL"], index=0
)

uploaded_file = None
url = None
if input_method == "Upload an Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
else:
    url = st.sidebar.text_input("Paste an image URL")

similarity_threshold = st.sidebar.slider("Similarity Threshold (%)", 0, 100, 25, 1)
search_button = st.sidebar.button("Find Similar Products", type="primary")

user_image = None
if uploaded_file:
    user_image = Image.open(uploaded_file)
elif url:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            user_image = Image.open(io.BytesIO(response.content))
    except Exception:
        pass

if user_image:
    st.header("Your Image")
    st.image(user_image, width=250)
    st.write("---")

if search_button and user_image:
    with st.spinner("Calculating similarities..."):
        user_embedding = model.encode(user_image)
        products_df['similarity'] = products_df['embedding'].apply(
            lambda emb: cosine_similarity(user_embedding, np.array(emb))
        )
        st.session_state.results = products_df.sort_values(by='similarity', ascending=False)

if st.session_state.results is not None:
    threshold_decimal = similarity_threshold / 100.0
    filtered_results = st.session_state.results[st.session_state.results['similarity'] >= threshold_decimal]

    st.header(f"Found {len(filtered_results)} Similar Products")

    if filtered_results.empty:
        st.warning("No products found above the selected similarity threshold. Try lowering the slider.")
    else:
        num_columns = 5
        for i in range(0, len(filtered_results), num_columns):
            chunk = filtered_results.iloc[i:i + num_columns]
            cols = st.columns(num_columns)

            for j, (index, row) in enumerate(chunk.iterrows()):
                with cols[j]:
                    image_path = f"public{row['url']}"
                    base64_image = image_to_base64(image_path)
                    st.markdown(f"""
                        <div class="product-card">
                            <div class="product-image-container">
                                <img src="data:image/jpeg;base64,{base64_image}" />
                            </div>
                            <div class="product-details">
                                <b>{row['name']}</b><br>
                                Similarity: <b>{row['similarity']:.2%}</b>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)