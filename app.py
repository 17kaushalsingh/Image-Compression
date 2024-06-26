# import streamlit as st
# from functions import load_image, compress_image, save_image
# import os

# def main():
#     st.title("Image Compressor using K-Means Clustering")

#     # Step 1: User inputs the file path
#     image_path = st.text_input("Enter the file path of the image:")
    
#     if image_path:
#         try:
#             # Step 2: Load the image and validate
#             image = load_image(image_path)
#             st.image(image, caption="Original Image", use_column_width=True)

#             # Step 3: Compress the image
#             n_colors = st.slider("Number of colors for compression:", 1, 64, 16)
#             compressed_image = compress_image(image, n_colors)
            
#             # Step 4: Display the compressed image
#             st.image(compressed_image, caption="Compressed Image", use_column_width=True)
            
#             # Step 5: Save and provide download link
#             output_path = "compressed_image.png"
#             save_image(compressed_image, output_path)
            
#             with open(output_path, "rb") as file:
#                 btn = st.download_button(label="Download Compressed Image", data=file, file_name=output_path, mime="image/png")

#         except Exception as e:
#             st.error(f"Error: {e}")

# if __name__ == "__main__":
#     main()


'''
import streamlit as st
from functions import load_image, compress_image, save_image
import os
from PIL import Image
import io

def main():
    st.title("Image Compressor using K-Means Clustering")

    # Step 1: User uploads the file
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Step 2: Load the image and validate
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)

            # Step 3: Compress the image
            n_colors = st.slider("Number of colors for compression:", 1, 64, 16)
            compressed_image = compress_image(image, n_colors)
            
            # Step 4: Display the compressed image
            st.image(compressed_image, caption="Compressed Image", use_column_width=True)
            
            # Step 5: Save and provide download link
            output = io.BytesIO()
            compressed_image.save(output, format="PNG")
            output.seek(0)

            st.download_button(label="Download Compressed Image", data=output, file_name="compressed_image.png", mime="image/png")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
'''

import streamlit as st
from functions import load_image, compress_image, save_image
from PIL import Image
import io

def main():
    st.title("Image Compressor using K-Means Clustering")

    # Step 1: User uploads the file
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Step 2: Load the image and validate
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)

            # Convert the image to a NumPy array for processing
            image_np = load_image(image)

            # Step 3: Compress the image
            n_colors = st.slider("Number of colors for compression:", 1, 64, 16)
            compressed_image = compress_image(image_np, n_colors)
            
            # Step 4: Display the compressed image
            st.image(compressed_image, caption="Compressed Image", use_column_width=True)
            
            # Step 5: Save and provide download link
            output = io.BytesIO()
            compressed_image.save(output, format="PNG")
            output.seek(0)

            st.download_button(label="Download Compressed Image", data=output, file_name="compressed_image.png", mime="image/png")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
