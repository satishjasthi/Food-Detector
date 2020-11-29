import io, torch
from pathlib import Path
import streamlit as st
from PIL import Image
from torchvision import transforms, models

ProjectRoot = Path(__file__).resolve().parent.parent
# load model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(str(ProjectRoot/'src/best_model.pth')))
model.eval()
classes = ['Not White Rice', 'White Rice']

def predict_image(image):
    test_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = test_transforms(image)
    print(image_tensor.shape)
    image_tensor = image_tensor.unsqueeze(0)

    
    output = model(image_tensor)
    print(output)
    return classes[output.data.numpy().argmax()]

st.title("Detectron")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    label = predict_image(image)
    st.markdown(f'<h2 align="center"> {label} </h2>', unsafe_allow_html=True)