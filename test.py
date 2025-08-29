# %% [markdown]
# ## Grab the train and test

# %%
import os
# import zipfile

# %%
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import dataloader,DataLoader
import matplotlib.pyplot as plt

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %% [markdown]
# #### Create Custom DataSet class

# %%
train_dir = "../split_dataset/train"
test_dir = "../split_dataset/test"


# %% [markdown]
# #### Data Transformation

# %%
# Transform for training dataset
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),               # Resize to 128x128
    transforms.ToTensor(),                       # Convert image to PyTorch tensor
])

# Transform for test dataset
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# %%
## Loading the dataset

train_dataset = ImageFolder(root=train_dir,transform=train_transform)
test_dataset = ImageFolder(root=test_dir,transform=test_transform)

# %%
# Create DataLoaders for train and test datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %% [markdown]
#  Build the CNN Model

# %%
class AlzheimerDetector(nn.Module):
  """
  Model Arch replicates TinyVGG model
  from CNN explainer website

  """

  def __init__(self, input_shape:int, hidden_units:int, output_shape:int,image_dimension:int):
    super().__init__()
    self.conv_block_1=nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_block_2=nn.Sequential(
    nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
    )

    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*image_dimension//2//2*image_dimension//2//2,
                  out_features=output_shape)
    )

  def forward(self,x):
    return self.classifier(self.conv_block_2(self.conv_block_1(x)))



# %%
torch.manual_seed(42)
model=AlzheimerDetector(input_shape=3,
                            hidden_units=10,
                            output_shape=4,
                            image_dimension=128).to(device)

model

# %%
## Load a sample image from train dataloader
sample_batch,sample_label=next(iter(train_loader))
sample_batch.shape

# %%
image = torch.randn(32,3,128,128)

conv_block_1=nn.Sequential(
        nn.Conv2d(in_channels=3,
                  out_channels=10,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=10,
                  out_channels=10,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

conv_block_2=nn.Sequential(
    nn.Conv2d(in_channels=10,
                  out_channels=10,
                  kernel_size=3,
                  stride=1,
                  padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=10,
                  out_channels=10,
                  kernel_size=3,
                  stride=1,
                  padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
    )
classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=10*32*32,
                  out_features=4)
    )

print(image.shape)
print(conv_block_1(image).shape)
print(conv_block_2(conv_block_1(image)).shape)
print(classifier(conv_block_2(conv_block_1(image))).shape)

# %% [markdown]
# ## Get Helper Functions

# %%
import requests
import os

if os.path.exists("helper_functions.py"):
  print("Skipping download, helper_functions.py exists!")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

# %%
from helper_functions import plot_predictions, plot_decision_boundary, accuracy_fn, print_train_time

# %% [markdown]
# ## Train-Test Loop time Functions

# %%
train_loss_arr=[]
test_loss_arr=[]

train_acc_arr=[]
test_acc_arr=[]

# %%
def train_step(model:nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device:torch.device,
               accuracy_fn:accuracy_fn):

  """Performs a training with model trying to learn on data_loader"""

  train_loss,train_acc=0,0
  model.train()



    ## Training
  train_loss=0 ## for calculating training_loss per batch

    ## Add a loop to loop through the training batches
  for batch,(X,y) in enumerate(dataloader):

    X,y=X.to(device),y.to(device)

      ## Forward pass
    y_pred=model(X)

      ## Calc the loss(per batch)
    loss=loss_fn(y_pred,y)
    train_loss+=loss.item()
    train_acc+=accuracy_fn(y,y_pred.argmax(dim=1))

      ## Optimizer zero grad
    optimizer.zero_grad()

      ## Loss backward
    loss.backward()

      ## Optimizer step
    optimizer.step()



  train_loss/=len(dataloader)
  train_acc/=len(dataloader)

  train_loss_arr.append(train_loss)
  train_acc_arr.append(train_acc)

  print(f"\n Train Loss: {train_loss:.4f}  | Train Accuracy: {train_acc}")





# %%
def test_step(model:nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               device:torch.device,
               accuracy_fn:accuracy_fn):

  """Performs a testing with model trying to learn on data_loader"""

  test_loss,test_acc=0,0
  model.eval()

  with torch.inference_mode():
    for batch,(X,y) in enumerate(dataloader):

      X,y=X.to(device),y.to(device)

      ## Forward pass
      y_pred=model(X)

      ## Calc the loss(per batch)
      loss=loss_fn(y_pred,y)
      test_loss+=loss.item()
      test_acc+=accuracy_fn(y,y_pred.argmax(dim=1))

    test_loss/=len(dataloader)
    test_acc/=len(dataloader)
    test_loss_arr.append(test_loss)
    test_acc_arr.append(test_acc)

  print(f"\n Test Loss: {test_loss:.4f}  | Test Accuracy: {test_acc}")



# %%
torch.manual_seed(42)

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn=torch.nn.Module,
               accuracy_fn=accuracy_fn):
  """
  Returns a dictionary containing the results of model predicting on data_loader.
  """

  loss,acc=0,0

  model.eval()
  with torch.inference_mode():

    for X,y in data_loader:

      X,y=X.to(device),y.to(device)

      ## Forward pass
      y_pred=model(X)
      ## Calc the loss(per batch)
      loss+=loss_fn(y_pred,y)
      ## Calc the accuracy(per batch)
      acc+=accuracy_fn(y,y_pred.argmax(dim=1))

    ## Adjust metrics to get average loss and accuracy per batch
    loss=loss/len(data_loader)
    acc/=len(data_loader)


  return {"model_name":model.__class__.__name__,
          "model_loss":loss.item(),
          "model_acc":acc}

# %% [markdown]
# ## Setting up Loss function and optimizer

# %%
## Train the CNN on our dataset

## Setup loss func, eval metrics, optimizer
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=model.parameters(),
                          lr=0.001)


# %%
torch.manual_seed(42)
torch.cuda.manual_seed(42)

from timeit import default_timer as timer
from tqdm.auto import tqdm ## For progress bar

epochs=1

train_time_on_device_start=timer()

for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch}\n---------")


  train_step(model=model,
             dataloader=train_loader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             device=device,
             accuracy_fn=accuracy_fn)

  test_step(model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=device,
            accuracy_fn=accuracy_fn)


train_time_on_device_end=timer()

total_train_time_on_device=print_train_time(start=train_time_on_device_start,
                                         end=train_time_on_device_end,
                                         device=str(next(model.parameters()).device))


print(f"Train time on {device}: {total_train_time_on_device:.3f} seconds")

# %%
## Using the train-test loss array plot the loss over epochs

epochs_range = range(1, len(train_loss_arr)+1)

plt.plot(epochs_range, train_loss_arr, label='Training Loss')
plt.plot(epochs_range, test_loss_arr, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# %%
## Model Performance on Train Data

eval_model(model=model,
           data_loader=test_loader,
           loss_fn=loss_fn,
           accuracy_fn=accuracy_fn)

# %%
## Model Performance on Test Data

eval_model(model=model,
           data_loader=train_loader,
           loss_fn=loss_fn,
           accuracy_fn=accuracy_fn)

# %% [markdown]
# ## Perform Model Metric Analysis

# %% [markdown]
# ## Inferencing and Confusion Matrix

# %%
!pip install -q torchmetrics

# %%
import mlxtend
import torchmetrics
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# %%
def make_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    y_preds = []
    y_true = []

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Making predictions..."):
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_logit = model(X)

            # Get prediction probabilities and predicted labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

            # Collect predictions and true labels
            y_preds.append(y_pred.cpu())
            y_true.append(y.cpu())

    # Combine all predictions and true labels into tensors
    y_preds = torch.cat(y_preds)
    y_true = torch.cat(y_true)

    return y_preds, y_true


# %% [markdown]
# ### Importing Deps for Confusion Matrix

# %%
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix


# %% [markdown]
# ### Train Data Confusion Matrix and Metrics

# %%
y_preds_train,y_true_train=make_predictions(model=model,
                 dataloader=train_loader)

# %%
confmat = ConfusionMatrix(num_classes=len(train_dataset.classes),
                          task="multiclass")

confmat_tensor=confmat(preds=y_preds_train,target=y_true_train)


# %%
## Plot the confusion matrix

fig,ax=plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                           figsize=(10,7),
                           class_names=train_dataset.classes)
plt.show()

# %%
## Metrics

precision = precision_score(y_true_train, y_preds_train, average='weighted')
recall = recall_score(y_true_train, y_preds_train, average='weighted')
f1 = f1_score(y_true_train, y_preds_train, average='weighted')
accuracy=accuracy_score(y_true_train, y_preds_train)

print(f"Precision for Training Data: {precision:.4f}")
print(f"Recall for Training Data: {recall:.4f}")
print(f"F1-score for Training Data: {f1:.4f}")
print(f"Accuracy for Training Data: {accuracy:.4f}")

# %% [markdown]
# ### Test Data Confusion Matrix

# %%
y_preds_test,y_true_test=make_predictions(model=model,
                 dataloader=test_loader)

# %%
confmat = ConfusionMatrix(num_classes=len(test_dataset.classes),
                          task="multiclass")

confmat_tensor=confmat(preds=y_preds_test,target=y_true_test)


# %%
## Plot the confusion matrix

fig,ax=plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                           figsize=(10,7),
                           class_names=train_dataset.classes)
plt.show()

# %%
## Metrics

precision = precision_score(y_true_test, y_preds_test, average='weighted')
recall = recall_score(y_true_test, y_preds_test, average='weighted')
f1 = f1_score(y_true_test, y_preds_test, average='weighted')
accuracy=accuracy_score(y_true_test, y_preds_test)


print(f"Precision for Testing Data: {precision:.4f}")
print(f"Recall for Testing Data: {recall:.4f}")
print(f"F1-score for Testing Data: {f1:.4f}")
print(f"Accuracy for Testing Data: {accuracy:.4f}")

# %% [markdown]
# ## Saving the Model

# %%
import os

# %%
os.makedirs(os.path.join(os.getcwd(),"models"),exist_ok=True)
MODEL_NAME="alz_CNN.pt"
MODEL_PATH=os.path.join(os.getcwd(),"models",MODEL_NAME)

# %%
print(f"Saving model to: {MODEL_PATH}")
torch.save(obj=model.state_dict(),f=MODEL_PATH)

# %% [markdown]
# ## Loading the Model

# %%
loaded_model=AlzheimerDetector(input_shape=3,hidden_units=10,output_shape=4,image_dimension=128).to(device)

# %%
loaded_model.load_state_dict(torch.load(f=MODEL_PATH, weights_only=True))

# %%



