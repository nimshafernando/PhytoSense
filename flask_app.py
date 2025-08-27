import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models and data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = []
class_names = []
phytochemical_data = {}

# Model architectures
def create_resnet50_model(num_classes):
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model

def create_mobilenetv2_model(num_classes):
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model

def create_efficientnet_b0_model(num_classes):
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model

# Image preprocessing
val_tf = transforms.Compose([
    transforms.Resize(288),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_models_and_data():
    """Load trained models and phytochemical data"""
    global models, class_names, phytochemical_data
    
    # Load metadata
    metadata_path = "metadata/model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        class_names = metadata["class_names"]
        num_classes = len(class_names)
        print(f"✅ Loaded metadata: {num_classes} classes")
    else:
        print("❌ Metadata file not found.")
        return False

    # Load trained models
    if num_classes > 0:
        try:
            model_paths = [
                ("models/resnet50_ensemble.pth", create_resnet50_model),
                ("models/mobilenetv2_ensemble.pth", create_mobilenetv2_model),
                ("models/efficientnet_b0_ensemble.pth", create_efficientnet_b0_model)
            ]
            
            for path, create_func in model_paths:
                if os.path.exists(path):
                    model = create_func(num_classes)
                    model.load_state_dict(torch.load(path, map_location=device))
                    model.to(device)
                    model.eval()
                    models.append(model)
                    print(f"✅ Loaded {os.path.basename(path)}")
                else:
                    print(f"❌ Model file not found: {path}")
                    
            print(f"Total models loaded: {len(models)}")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

    # Load phytochemical mapping
    phyto_path = "data/phytochemical_mapping.json"
    if os.path.exists(phyto_path):
        try:
            with open(phyto_path, 'r') as f:
                phytochemical_data = json.load(f)
            print(f"✅ Loaded phytochemical data for {len(phytochemical_data)} plants")
        except Exception as e:
            print(f"❌ Error loading phytochemical mapping: {e}")
            return False
    else:
        print(f"❌ Phytochemical mapping file not found: {phyto_path}")
        return False
    
    return True

def smiles_to_image_base64(smiles):
    """Convert SMILES to base64 encoded 2D molecular structure image"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate 2D structure image
        img = Draw.MolToImage(mol, size=(400, 300))
        
        # Convert PIL image to base64 string
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error generating molecular image: {e}")
        return None

def generate_3d_mol_block(smiles):
    """Generate 3D MOL block from SMILES for 3Dmol.js"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        
        # Convert to mol block
        mol_block = Chem.MolToMolBlock(mol)
        return mol_block
        
    except Exception as e:
        print(f"Error generating 3D mol block: {e}")
        return None

def predict_leaf(image_path):
    """Predict leaf species using ensemble of models"""
    if not models:
        return "Models not loaded.", None, []

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    img_tensor = val_tf(image).unsqueeze(0).to(device)
    
    all_probs = []
    pred_rows = []

    # Get predictions from all models
    for i, model in enumerate(models):
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            pred_class = class_names[np.argmax(probs)]
            pred_conf = float(np.max(probs))  # Convert to Python float
            
            model_name = ['ResNet50', 'MobileNetV2', 'EfficientNet-B0'][i] if i < 3 else f'Model_{i+1}'
            pred_rows.append({
                "Model": model_name,
                "Predicted Class": pred_class,
                "Confidence (%)": round(pred_conf * 100, 2)
            })

    # Ensemble prediction
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_idx = np.argmax(ensemble_probs)
    ensemble_class = class_names[ensemble_idx]
    ensemble_conf = float(ensemble_probs[ensemble_idx])  # Convert to Python float
    pred_rows.append({
        "Model": "Ensemble",
        "Predicted Class": ensemble_class,
        "Confidence (%)": round(ensemble_conf * 100, 2)
    })

    # Match phytochemical data
    def normalize(k): 
        return k.strip().lower().replace(" ", "").replace("_", "")
    
    norm_target = normalize(ensemble_class)
    match_key = next((k for k in phytochemical_data if normalize(k) == norm_target), None)

    if match_key:
        plant_info = phytochemical_data[match_key]
        phytochemicals = plant_info.get("phytochemicals", [])
        common_name = plant_info.get("common_name", "N/A")
    else:
        phytochemicals = []
        common_name = "N/A"

    result = {
        "prediction": ensemble_class,
        "common_name": common_name,
        "confidence": round(float(ensemble_conf) * 100, 2),  # Ensure it's a Python float
        "model_results": pred_rows
    }

    return result, phytochemicals

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result, phytochemicals = predict_leaf(filepath)
            
            # Process phytochemicals for frontend
            processed_compounds = []
            for compound in phytochemicals:
                name = compound.get("name", "Unknown")
                description = compound.get("description", "No description available")
                smiles = compound.get("smiles", "")
                
                compound_data = {
                    "name": name,
                    "description": description,
                    "smiles": smiles,
                    "image_2d": None,
                    "mol_block_3d": None
                }
                
                if smiles:
                    # Generate 2D image
                    compound_data["image_2d"] = smiles_to_image_base64(smiles)
                    # Generate 3D mol block
                    compound_data["mol_block_3d"] = generate_3d_mol_block(smiles)
                
                processed_compounds.append(compound_data)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'result': result,
                'compounds': processed_compounds
            })
            
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask Medicinal Leaf Classifier with 3D Molecular Visualization...")
    
    # Load models and data
    if load_models_and_data():
        print("✅ All models and data loaded successfully!")
        print("🌐 Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load models and data. Please check your files.")
