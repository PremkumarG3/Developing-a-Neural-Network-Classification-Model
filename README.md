# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="741" height="858" alt="Screenshot 2026-02-10 112103" src="https://github.com/user-attachments/assets/af667028-b791-44ff-a02a-efe19f6aab3a" />

## DESIGN STEPS
### Step 1: Load and Preprocess Data
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### Step 2: Feature Scaling and Data Split
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### Step 3: Convert Data to PyTorch Tensors
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### Step 4: Define the Neural Network Model
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### Step 5: Train the Model
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### Step 6: Evaluate and Predict
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.

## PROGRAM

### Name:Prem Kumar G

### Register Number:212223230158

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)

    def forward(self, x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x

# Initialize the Model, Loss Function, and Optimizer
# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
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

# Initialize model
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
train_model(model,train_loader,criterion,optimizer,epochs=100)
```

### Dataset Information
<img width="1384" height="235" alt="Screenshot 2026-02-10 110923" src="https://github.com/user-attachments/assets/c8fb0f0a-b5d9-449c-90ea-f43da0aa0458" />

### OUTPUT

## Confusion Matrix

<img width="1400" height="537" alt="Screenshot 2026-02-10 110940" src="https://github.com/user-attachments/assets/5a8e7f08-55c0-417a-b87e-7436aa0da363" />

## Classification Report
<img width="1392" height="401" alt="Screenshot 2026-02-10 110953" src="https://github.com/user-attachments/assets/3b3f374d-09c2-4113-9508-df965dec37c8" />

### New Sample Data Prediction
<img width="1388" height="93" alt="Screenshot 2026-02-10 111003" src="https://github.com/user-attachments/assets/d4d4c514-c6af-4ed0-a187-dbe563c21429" />

## RESULT
Thus, the program was executed successfully.
