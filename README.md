# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="815" height="854" alt="image" src="https://github.com/user-attachments/assets/4ae3309c-4d35-4880-8c17-978d9404a26f" />


## DESIGN STEPS
### STEP 1:

Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2:

Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### STEP 3:

Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### STEP 4:

Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5:

Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6:

Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.

## PROGRAM

### Name: chandru k

### Register Number: 212224220017

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1 =nn.Linear(input_size,32)
        self.fc2 =nn.Linear(32, 16)
        self.fc3 =nn.Linear(16, 8)
        self.fc4 =nn.Linear(8, 4)
    def forward(self, x):
      #Include your code here
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x


        
# Initialize the Model, Loss Function, and Optimizer

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
  #Include your code here
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
  if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
train_model(model, train_loader, criterion, optimizer, epochs=100)

```

### Dataset Information
<img width="1390" height="257" alt="image" src="https://github.com/user-attachments/assets/b389c821-1e98-4371-ad67-6d2884493abf" />

### OUTPUT
<img width="615" height="137" alt="image" src="https://github.com/user-attachments/assets/58336454-5d6b-4687-a35f-cf7d6a44ddbd" />

## Confusion Matrix

<img width="795" height="601" alt="image" src="https://github.com/user-attachments/assets/4e9539a4-d449-4bdc-9b9b-aeae1b10b0fd" />



## Classification Report

<img width="719" height="462" alt="image" src="https://github.com/user-attachments/assets/97d37bf0-0f22-47d6-997b-27bdc528245f" />



### New Sample Data Prediction

<img width="615" height="137" alt="image" src="https://github.com/user-attachments/assets/83249afb-173a-4c45-a74f-67a4dbb1985c" />



## RESULT

 A neural network classification model was successfully developed and tested on the given dataset with satisfactory classification performance.

