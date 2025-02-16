# app.py
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import torch.nn as nn

class DogBreedClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DogBreedClassifier, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)
        
    def forward(self, x):
        return self.vit(x)

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model():
    # Load label mapping
    mapping = torch.load('label_mapping.pth', map_location='cpu')
    label_to_idx = mapping['label_to_idx']
    breeds = mapping['breeds']
    
    # Load model
    model = DogBreedClassifier(num_classes=len(breeds))
    checkpoint = torch.load('best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, breeds

def main():
    st.title("Dog Breed Classification")
    
    try:
        model, breeds = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure 'best_model.pth' and 'label_mapping.pth' are in the same directory.")
        return
    
    uploaded_file = st.file_uploader("Choose a dog image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make prediction
        transform = get_transform()
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            
            # Display predictions
            st.write("### Top 3 Predictions:")
            for prob, idx in zip(top3_prob, top3_idx):
                breed = breeds[idx.item()]
                st.write(f"{breed}: {prob.item()*100:.2f}%")
                
                # Create a progress bar for visualization
                st.progress(prob.item())

if __name__ == "__main__":
    main()
