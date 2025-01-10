import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = Image.open(test_image).resize((64, 64))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # convert single image to batch
    predictions = model.predict(input_arr)
    max_index = np.argmax(predictions)  # index of the max element
    max_prob = predictions[0][max_index]  # probability of the max element
    return max_index, max_prob  # return index and probability

# Styling
st.set_page_config(
    page_title="Fruits & Vegetables Recognition",
    layout="wide",  # Use wide layout for better spacing
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.title("🍎 Fruits & Vegetables Recognition System 🍇")
    
    # Menambahkan path gambar
    image_path = "Home.jpeg"
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
Selamat datang di Sistem Pengenalan Buah dan Sayur!
Aplikasi ini menggunakan model Deep learning untuk mengidentifikasi buah dan sayuran dari gambar.
            """
        )
    with col2:
        # Menampilkan gambar dari path dengan penyesuaian lebar
        st.image(image_path, use_column_width=True)


elif app_mode == "About Project":
    st.title("📘 About the Project")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
            This project is based on a dataset containing images of fruits and vegetables.  
            The dataset includes 36 categories with a total of 2900 images.  
            **Categories include:**  
            - **Fruits:** Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, etc.  
            - **Vegetables:** Cucumber, Carrot, Onion, Potato, Tomato, Spinach, etc.
        """)
    with col2:
        st.image("about.jpg", caption="Sample Dataset", use_column_width=True)

elif app_mode == "Prediction":
    st.title("🔍 Model Prediction")
    st.markdown("Upload an image of a fruit or vegetable, and the model will predict its category.")
    
    # Layout for image upload and prediction
    col1, col2 = st.columns([2, 1])

    # Image upload
    with col1:
        test_image = st.file_uploader(
            "Upload an Image:", type=["jpg", "png", "jpeg"], accept_multiple_files=False
        )

    if test_image:
        # Open and resize the image for display
        uploaded_image = Image.open(test_image)
        resized_image = uploaded_image.resize((300, 300), Image.Resampling.LANCZOS)  # Resize with smooth scaling
        st.image(resized_image, caption="Uploaded Image (Resized for Display)", width=250)  # Set explicit width

        if st.button("🔮 Predict", key="predict_button"):
            with st.spinner("Predicting... Please wait..."):
                result_index, confidence = model_prediction(test_image)
                
                # Labels
                label = [
                    "apple", "banana", "beetroot", "bell pepper", "cabbage",
                    "capsicum", "carrot", "cauliflower", "chilli pepper", "corn",
                    "cucumber", "eggplant", "garlic", "ginger", "grapes", 
                    "jalepeno", "kiwi", "lemon", "lettuce", "mango",
                    "onion", "orange", "paprika", "pear", "peas",
                    "pineapple", "pomegranate", "potato", "raddish", "soy beans",
                    "spinach", "sweetcorn", "sweetpotato", "tomato", "turnip",
                    "watermelon"
                ]
                
                # Display Results
                predicted_label = label[result_index]
                
                st.markdown(
                    f"""
                    ### 🎯 Prediction: **{predicted_label.capitalize()}**
                    - **Confidence Level:** {confidence:.2%}
                    """
                )
                st.progress(int(confidence * 100))  # Convert confidence to percentage
                
                # Vitamins and Category
                vitamin_dict = {
                    "apple": "Vitamin C, Vitamin K",
                    "banana": "Vitamin B6, Vitamin C",
                    "beetroot": "Folate (Vitamin B9), Vitamin C",
                    "bell pepper": "Vitamin C, Vitamin A",
                    "cabbage": "Vitamin K, Vitamin C",
                    "capsicum": "Vitamin C, Vitamin A",
                    "carrot": "Vitamin A, Vitamin K",
                    "cauliflower": "Vitamin C, Vitamin K",
                    "chilli pepper": "Vitamin C, Vitamin B6",
                    "corn": "Vitamin B, Vitamin C",
                    "cucumber": "Vitamin K, Vitamin C",
                    "eggplant": "Vitamin K, Vitamin B6",
                    "garlic": "Vitamin B6, Vitamin C",
                    "ginger": "Vitamin B6, Vitamin C",
                    "grapes": "Vitamin C, Vitamin K",
                    "jalepeno": "Vitamin C, Vitamin B6",
                    "kiwi": "Vitamin C, Vitamin E",
                    "lemon": "Vitamin C, Vitamin B6",
                    "lettuce": "Vitamin K, Vitamin A",
                    "mango": "Vitamin C, Vitamin A",
                    "onion": "Vitamin C, Vitamin B6",
                    "orange": "Vitamin C, Folate",
                    "paprika": "Vitamin C, Vitamin A",
                    "pear": "Vitamin C, Vitamin K",
                    "peas": "Vitamin A, Vitamin K",
                    "pineapple": "Vitamin C, Vitamin B6",
                    "pomegranate": "Vitamin C, Folate",
                    "potato": "Vitamin C, Vitamin B6",
                    "raddish": "Vitamin C, Folate",
                    "soy beans": "Vitamin C, Vitamin K",
                    "spinach": "Vitamin A, Vitamin C",
                    "sweetcorn": "Vitamin B, Vitamin C",
                    "sweetpotato": "Vitamin A, Vitamin C",
                    "tomato": "Vitamin C, Vitamin K",
                    "turnip": "Vitamin C, Vitamin B6",
                    "watermelon": "Vitamin C, Vitamin A"
                }
                category_dict = {
                    "apple": "Fruit", "banana": "Fruit", "beetroot": "Vegetable", 
                    "bell pepper": "Vegetable", "cabbage": "Vegetable", 
                    "capsicum": "Vegetable", "carrot": "Vegetable", 
                    "cauliflower": "Vegetable", "chilli pepper": "Vegetable", 
                    "corn": "Vegetable", "cucumber": "Vegetable", "eggplant": "Vegetable", 
                    "garlic": "Vegetable", "ginger": "Vegetable", "grapes": "Fruit", 
                    "jalepeno": "Vegetable", "kiwi": "Fruit", "lemon": "Fruit", 
                    "lettuce": "Vegetable", "mango": "Fruit", "onion": "Vegetable", 
                    "orange": "Fruit", "paprika": "Vegetable", "pear": "Fruit", 
                    "peas": "Vegetable", "pineapple": "Fruit", "pomegranate": "Fruit", 
                    "potato": "Vegetable", "raddish": "Vegetable", "soy beans": "Vegetable", 
                    "spinach": "Vegetable", "sweetcorn": "Vegetable", "sweetpotato": "Vegetable", 
                    "tomato": "Vegetable", "turnip": "Vegetable", "watermelon": "Fruit"
                }
                
                # Display vitamins and category
                col3, col4 = st.columns(2)
                with col3:
                    st.info(f"**Category:** {category_dict.get(predicted_label, 'Unknown')}")
                with col4:
                    st.info(f"**Vitamins:** {vitamin_dict.get(predicted_label, 'No data available')}")
