import torch
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTForImageClassification
# Define the parameters
image_size = 224
num_classes = 8
# Initialize the VisionTransformer model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_classes)

# Load the state_dict into the model
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
# Function to perform image classification
def classify_image(image):




    # Set the model to evaluation mode
    model.eval()

    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize image to (224, 224)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the uploaded image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Pass the input through the model
    with torch.no_grad():
        output = model(input_batch)

    logits = output.logits
    predicted_class_index = logits.argmax(dim=1).item()

    # Assuming you have a list of class labels
    class_labels = [ 'Artifacts', 'Games','Nutrition' , 'Fashion','Accessories', 'Beauty', 'Home', 'Stationary']

    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label

def main():


    # Add your logo image in the corner
    st.sidebar.image("slash_creators_logo.png", use_column_width=True)

    # Title and upload option
    st.title("Image Classification")
    st.write("Upload an image for classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform classification
        predicted_class = classify_image(image)
        st.write(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
