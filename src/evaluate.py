import os
import torch
from dataset import get_dataloader
from model import FaceShapeModel
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def evaluate(data_dir, model_name, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dir = os.path.join(data_dir, "testing_set")
    if not os.path.exists(test_dir):
        print(f"Directory {test_dir} not found.")
        return

    val_loader, classes = get_dataloader(test_dir, batch_size=32, is_train=False)
    num_classes = len(classes)
    
    model = FaceShapeModel(num_classes=num_classes, model_name=model_name)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    os.makedirs("../outputs", exist_ok=True)
    plt.savefig("../outputs/confusion_matrix.png")
    print("Saved confusion matrix to outputs/confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/face-shape-dataset/FaceShape Dataset")
    parser.add_argument("--model", type=str, default="mobilenet_v3_small")
    parser.add_argument("--weights", type=str, default="../outputs/best_mobilenet_v3_small.pth")
    args = parser.parse_args()
    
    evaluate(args.data_dir, args.model, args.weights)
