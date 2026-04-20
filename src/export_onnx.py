import torch
import argparse
from model import FaceShapeModel

def export_to_onnx(model_name, weights_path, output_path, num_classes=5):
    # Initialize the model on CPU
    model = FaceShapeModel(num_classes=num_classes, model_name=model_name)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    # Dummy input for tracing (Batch Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 150, 150, device='cpu')

    print(f"Exporting model {model_name} to ONNX format...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Successfully exported to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet_v2_s")
    parser.add_argument("--weights", type=str, default="../outputs/best_efficientnet_v2_s.pth")
    parser.add_argument("--output", type=str, default="../outputs/face_shape_model.onnx")
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.weights, args.output)
