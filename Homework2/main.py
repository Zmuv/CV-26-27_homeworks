import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}\n")


def sync():
    # GPU work runs asynchronously: wait for it to finish before reading the clock
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

NUM_EPOCHS = 15


# Load MobileNet V3, pre-trained on ImageNet-1K
weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights).to(device)

# Training set transformations with augmentations
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_tf = weights.transforms()

# This augmented dataset will be used for training
train_dataset_augmented = datasets.OxfordIIITPet(root="./dataset", split="trainval", download=True, transform=train_tf)
# This plain dataset will be used for the validation set
train_dataset_normal = datasets.OxfordIIITPet(root="./dataset", split="trainval", download=True, transform=test_tf)
test_dataset = datasets.OxfordIIITPet(root="./dataset", split="test", download=True, transform=test_tf)

# Split the image indices 80/20 stratified
indices = np.arange(len(train_dataset_normal))
train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=train_dataset_normal._labels, random_state=42)

# Keep augmented images of those 80% indices
train_dataset = Subset(train_dataset_augmented,train_idx)
# Keep non-augmented images the other 20% indices
val_dataset = Subset(train_dataset_normal, val_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --- Full fine-tuning ---
# Substitute the final layer of the classifier with a new 37-output layer
num_features = model.classifier[3].in_features # Number of input features
NUM_CLASSES = 37
model.classifier[3] = nn.Linear(num_features, NUM_CLASSES).to(device) # Set new final layer

# Set up the optimizer
optimizer = optim.Adam(params=model.parameters(),lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Print the number of total and trainable parameters
total_params= sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

# Run training
print("Training complete model with last layer changed")
sync()
best_val_accuracy, best_state = 0.0, None
start = time.perf_counter()
for EPOCH in range (NUM_EPOCHS):
  model.train()
  running_loss = 0.0
  print(f"epoch: {EPOCH}")  
  for step, (images, labels) in enumerate(train_loader):
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if (step + 1) % 50 == 0:
          print(f"Step [{step + 1}/{len(train_loader)}], Loss: {running_loss / 50:.4f}")
          running_loss = 0.0  
  # validation after each epoch
  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Predicted class is index of highest logit along dimension 1
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    val_accuracy = 100.0 * correct / total
    print(f"Validation accuracy: {val_accuracy:.2f}% ({correct}/{total})")
    # Keep the best checkpoint
    if(val_accuracy > best_val_accuracy):
      best_val_accuracy = val_accuracy
      best_state = copy.deepcopy(model.state_dict())
sync()
train_time_finetune = time.perf_counter() - start

# Load the best checkpoint
model.load_state_dict(best_state)
# Evaluate top-1 accuracy on the test set
model.eval()
correct = 0
total = 0

# Save the true and predicted labes used later for confusion matrices
y_gt_finetune, y_pred_finetune = [], []

start = time.perf_counter()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Predicted class is index of highest logit along dimension 1
        _, predicted = torch.max(outputs, 1)
        
        y_pred_finetune.append(predicted.cpu().numpy())
        y_gt_finetune.append(labels.cpu().numpy())
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
eval_time_finetune = time.perf_counter() - start

y_gt_finetune = np.concatenate(y_gt_finetune)
y_pred_finetune = np.concatenate(y_pred_finetune)

accuracy = 100.0 * correct / total
print(f"Test Set Top-1 Accuracy: {accuracy:.2f}% ({correct}/{total})")
print(f"Training time: {train_time_finetune:.1f} s | Test evaluation time: {eval_time_finetune:.1f} s")

# --- CLASSIFIER HEAD TRAINING ---
# load a fresh model
model = models.mobilenet_v3_small(weights=weights).to(device)

# Freeze the convolutional layers
for param in model.features.parameters():
    param.requires_grad = False

# Substitute the last layer with a 37-outputs layer
num_features = model.classifier[3].in_features # Number of input features
NUM_CLASSES = 37
model.classifier[3] = nn.Linear(num_features, NUM_CLASSES).to(device) # Set new final layer

# Print the number of total and trainable parameters
total_params= sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

# Optimize only the fully connected layers (1st and last)
optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Run training
print("\nTraining model with frozen convolutional layer and last layer changed")
sync()
best_val_accuracy, best_state = 0.0, None
start = time.perf_counter()
for EPOCH in range (NUM_EPOCHS):
  print(f"epoch: {EPOCH}") 
  model.train()
  # This ensures the backbone is completely frozen, some layers do not get frozen with requires_grad = False
  model.features.eval()
  running_loss = 0.0 
  for step, (images, labels) in enumerate(train_loader):
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if (step + 1) % 50 == 0:
          print(f"Step [{step + 1}/{len(train_loader)}], Loss: {running_loss / 50:.4f}")
          running_loss = 0.0
  # validation after each epoch
  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Predicted class is index of highest logit along dimension 1
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    val_accuracy = 100.0 * correct / total
    print(f"Validation accuracy: {val_accuracy:.2f}% ({correct}/{total})")
    # Keep only the best checkpoint
    if(val_accuracy > best_val_accuracy):
      best_val_accuracy = val_accuracy
      best_state = copy.deepcopy(model.state_dict())
sync()
train_time_head = time.perf_counter() - start

# Load the best epoch checkpoint
model.load_state_dict(best_state)

# Evaluate top-1 accuracy on the test set
model.eval()
correct = 0
total = 0

# Keep the true and predicted labels that will be used later for the confusion matrices
y_gt_head, y_pred_head = [], []

start = time.perf_counter()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Predicted class is index of highest logit along dimension 1
        _, predicted = torch.max(outputs, 1)

        y_pred_head.append(predicted.cpu().numpy())
        y_gt_head.append(labels.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
eval_time_head = time.perf_counter() - start

y_gt_head = np.concatenate(y_gt_head)
y_pred_head = np.concatenate(y_pred_head)

accuracy = 100.0 * correct / total
print(f"Test Set Top-1 Accuracy: {accuracy:.2f}% ({correct}/{total})")
print(f"Training time: {train_time_head:.1f} s | Test evaluation time: {eval_time_head:.1f} s")


# --- Feature extraction with a traditional classifier ---
# load a fresh model
model = models.mobilenet_v3_small(weights=weights).to(device)

# Freeze the convolutional layers (Not really needed here)
for param in model.features.parameters():
    param.requires_grad = False

# Only keep the convolutional and avgpool layers
model.classifier = nn.Identity()

# Set model in evaluation mode 
model.eval()

x_train_logistic, y_train_logistic = [], []


start = time.perf_counter()
# Turn the dataset into a list of value / labels where the value is the output of the convolutional feature extractor
with torch.no_grad():
  # This for is used to cycle through the augmentations
  for _ in range(5):
    for images, labels in train_loader:
        x_train_logistic.append(model(images.to(device)).cpu().numpy())
        y_train_logistic.append(labels.numpy())
x_train_logistic = np.concatenate(x_train_logistic)
y_train_logistic = np.concatenate(y_train_logistic)

x_val_logistic, y_val_logistic = [], []
with torch.no_grad():
  for images, labels in val_loader:
    x_val_logistic.append(model(images.to(device)).cpu().numpy())
    y_val_logistic.append(labels.numpy())
x_val_logistic = np.concatenate(x_val_logistic)
y_val_logistic = np.concatenate(y_val_logistic)
    
x_test_logistic, y_test_logistic = [], []
with torch.no_grad():
  for images, labels in test_loader:
    x_test_logistic.append(model(images.to(device)).cpu().numpy())
    y_test_logistic.append(labels.numpy())

x_test_logistic = np.concatenate(x_test_logistic)
y_test_logistic = np.concatenate(y_test_logistic)
sync()
extract_time = time.perf_counter() - start

print("\nFitting logistic regression")

# Test different C values to find the best one
C_values = [0.001, 0.01, 0.1, 1, 10]
start = time.perf_counter()
best_val_accuracy, best_c, best_logreg = 0.0, 0.0001, None
for c in C_values:
  logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42, C=c))
  logreg.fit(x_train_logistic, y_train_logistic)
  # Check the accuracy after every C value change
  val_accuracy = 100 * logreg.score(x_val_logistic, y_val_logistic)
  print(f"Validation accuracy: {val_accuracy}%")
  # Keep the best model
  if(val_accuracy > best_val_accuracy):
    best_val_accuracy = val_accuracy
    best_c = c
    best_logreg = logreg
fit_time = time.perf_counter() - start

print(f"Selected C={best_c} (val acc {best_val_accuracy:.2f}%)")
print(f"Top-1 accuracy: {100 * best_logreg.score(x_test_logistic, y_test_logistic):.2f}%")
print(f"Feature extraction time (train + test): {extract_time:.1f} s | Logistic regression fit time: {fit_time:.1f} s")

# Get prediction labels for the logistic regression
y_pred = best_logreg.predict(x_test_logistic)

# Macro F1: F1 computed per breed and then averaged
f1_finetune = metrics.f1_score(y_gt_finetune, y_pred_finetune, average="macro")
f1_head = metrics.f1_score(y_gt_head, y_pred_head, average="macro")
f1_logreg = metrics.f1_score(y_test_logistic, y_pred, average="macro")

print("\nMacro F1 score")
print(f"Full fine-tuning: {f1_finetune:.4f}")
print(f"Classifier head: {f1_head:.4f}")
print(f"Features + logistic regression: {f1_logreg:.4f}")

# Confusion matrix with the 37 names
cnf_matrix = metrics.confusion_matrix(y_true=y_test_logistic, y_pred=y_pred)
cnf_matrix_finetune = metrics.confusion_matrix(y_true=y_gt_finetune, y_pred=y_pred_finetune)
cnf_matrix_head = metrics.confusion_matrix(y_true=y_gt_head, y_pred=y_pred_head)
class_names = test_dataset.classes
fig, ax = plt.subplots(figsize=(16, 14))
ConfusionMatrixDisplay(cnf_matrix, display_labels=test_dataset.classes).plot(
    ax=ax, cmap="YlGnBu", xticks_rotation=90, values_format="d", text_kw={"size": 6})
ax.set_title("Confusion matrix (logistic regression on MobileNetV3 features)")
plt.tight_layout()
plt.savefig("confusion_matrix_logreg.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(16, 14))
ConfusionMatrixDisplay(cnf_matrix_finetune, display_labels=test_dataset.classes).plot(
    ax=ax, cmap="YlGnBu", xticks_rotation=90, values_format="d", text_kw={"size": 6})
ax.set_title("Confusion matrix finetuning of MobileNetV3")
plt.tight_layout()
plt.savefig("confusion_matrix_finetune.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(16, 14))
ConfusionMatrixDisplay(cnf_matrix_head, display_labels=test_dataset.classes).plot(
    ax=ax, cmap="YlGnBu", xticks_rotation=90, values_format="d", text_kw={"size": 6})
ax.set_title("Confusion matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_classifier_head.png", dpi=150)
plt.show()