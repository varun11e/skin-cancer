import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image

# Class labels
CLASS_LABELS = ['NORMAL_SKIN', 'PSORIASIS', 'Ringworm', 'acne']

def train_model(data_dir, epochs=10, batch_size=16, lr=0.001, save_path='skin_disease_model.pth'):
    print(f"Initializing training using dataset from: {data_dir}")
    
    # 1. Setup transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # 2. Check if train/val subdirectories exist
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_path):
        print(f"Error: Train subdirectory not found at: {train_path}")
        print("Please structure your dataset folder as:")
        print("  dataset/")
        print("    ├── train/")
        print("    │    ├── NORMAL_SKIN/")
        print("    │    ├── PSORIASIS/")
        print("    │    ├── Ringworm/")
        print("    │    └── acne/")
        print("    └── val/")
        print("         ├── NORMAL_SKIN/")
        print("         ├── PSORIASIS/")
        print("         ├── Ringworm/")
        print("         └── acne/")
        return
        
    # If val subdirectory does not exist, we will use validation split from train
    if not os.path.exists(val_path):
        print(f"Validation directory not found. Using train folder as validation set.")
        val_path = train_path

    # Load datasets
    try:
        image_datasets = {
            'train': datasets.ImageFolder(train_path, data_transforms['train']),
            'val': datasets.ImageFolder(val_path, data_transforms['val'])
        }
    except Exception as e:
        print(f"Error loading ImageFolder dataset: {e}")
        return

    # Check classes
    dataset_classes = image_datasets['train'].classes
    print(f"Detected classes in dataset: {dataset_classes}")
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    # 3. Choose device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 4. Load Pretrained EfficientNet
    print("Loading pretrained EfficientNet-B3...")
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    
    # Modify classifier to match 4 output classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 4)
    )
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Fine-tune only classifier or entire model (since small dataset, classifier is safer)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_acc = 0.0
    
    # 5. Training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward & optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save best model weight dict
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f"✓ Saved new best model to {save_path} with Accuracy: {best_acc:.4f}")
                
        print()
        
    print(f'Training complete. Best Validation Accuracy: {best_acc:4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DermaAI Skin Disease Classifier Model")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to your dataset folder containing train/val folders")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--save_path', type=str, default='skin_disease_model.pth', help="Where to save the trained model .pth file")
    
    args = parser.parse_args()
    
    # Run training
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )
