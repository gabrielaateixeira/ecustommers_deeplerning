# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import sklearn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('dark_background')

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/DeepLearning/Assign_01

data = pd.read_csv("Ecommerce Customers_Kaggle.csv")

len(data)

data.head()

data.info()

data.describe()

data.isnull().sum()

data1= data.drop(['Email', 'Address', 'Avatar'], axis=1)
plt.figure(figsize= (15, 5))
sns.heatmap(data1.corr(), annot= True, cmap='YlGnBu')
plt.show()

target_column = 'Yearly Amount Spent'

y = data[target_column]
y

# Just numeric colunms
X_numeric = data.select_dtypes(include=['number'])
X_numeric

# Calculate correlation with target
correlation = X_numeric.corrwith(y)

#Select top 4
top_features = correlation.abs().sort_values(ascending=False).head(4).index.tolist()

print("Top 4 features", top_features)

X = X_numeric[top_features]
X

y=y.to_numpy().reshape(-1,1)

fig,axes = plt.subplots(nrows=1,ncols=4,figsize=(16,6))
plt.suptitle('Inspecting Each Feature', fontsize = 30)

axes[0].plot(data['Length of Membership'],data['Yearly Amount Spent'],'mo')
axes[0].set_ylabel("Yearly Amount Spent", fontsize = 15)
axes[0].set_xlabel("Length of Membership", fontsize = 15)

axes[1].plot(data['Time on App'],data['Yearly Amount Spent'],'go')
axes[1].set_ylabel("Yearly Amount Spent", fontsize = 15)
axes[1].set_xlabel("Time on App", fontsize = 15)

axes[2].plot(data['Avg. Session Length'],data['Yearly Amount Spent'],'bo')
axes[2].set_ylabel("Yearly Amount Spent", fontsize = 15)
axes[2].set_xlabel("Avg. Session Length", fontsize = 15)

axes[3].plot(data['Time on Website'],data['Yearly Amount Spent'],'bo')
axes[3].set_ylabel("Yearly Amount Spent", fontsize = 15)
axes[3].set_xlabel("Time on Website", fontsize = 15)

plt.tight_layout();

"""Based on these plots, the lenght of membership looks to be the most correlated feature with Yearly Amount Spent"""

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=data)

X_new = X.iloc[:,:].values

X_new

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

X_test.shape

X_train.shape

y_train.shape

X_train

"""**Data Standardization**"""

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""# **Create Neural Network for Multiple Regression using TensorFlow**"""



# Test different architectures
def test_architectures(X_train, X_test, y_train, y_test, architectures, epochs=400):
    results = []

    for i, arch in enumerate(architectures):
        print(f"\n🔁 Testing architecture {i+1}: {arch}")


        # Tensorflow model
        model = Sequential()
        model.add(Dense(arch[0], activation='relu', input_shape=(X_train.shape[1],)))
        for units in arch[1:]:
            model.add(Dense(units, activation='relu'))
        model.add(Dense(1))  # Output

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=epochs, verbose=0)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)

        print(f"✅ RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        results.append({
            'architecture': arch,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })

    return results

# Architectures to test
architectures = [
    [4, 4],
    [8, 8],
    [8, 16, 8],
    [16, 32, 16],
    [16, 16, 16, 16]
]

# Test architectures
results = test_architectures(X_train, X_test, y_train, y_test, architectures, epochs=400)

# Display results
results_df = pd.DataFrame(results)
print("\n📊 Compare final results:")
print(results_df.sort_values(by='rmse'))

"""Best Architecture is [16, 16, 16, 16]"""

#Best architecture
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))  # output

model.summary()

model.compile(optimizer = 'adam' ,loss='mse')

history = model.fit(X_train,y_train, epochs = 400)

plt.figure(figsize =(12,6))
plt.plot(history.history['loss'],'m', lw = 4, label= 'Training Loss')
plt.xlabel('Epochs', fontsize = 15)
plt.ylabel( 'Loss', fontsize = 15)
plt.show()

print("Weights are :" )
print(model.layers[0].get_weights()[0])

print(" ")

print("Bias is :" )
print(model.layers[0].get_weights()[1])

"""Testing and evaluating the model"""

y_pred = model.predict(X_test)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('MAE:', np.sqrt(metrics.mean_absolute_error(y_test, y_pred)))
print('R2:', metrics.r2_score(y_test, y_pred))

"""Good perfomance, highly accurate. The low RMSE and MAE values further confirm that the model's predictions are very close to the actual values.

Further Evaluating the model
"""

test_data = X_test[99,:].reshape(1,4)
test_data.shape

test_label = y_test[99]
test_label

y_t = model.predict(test_data)

print('RMSE:', np.sqrt(metrics.mean_squared_error(test_label, y_t)))
print('MAE:', np.sqrt(metrics.mean_absolute_error(test_label, y_t)))
print('R2:', metrics.r2_score(test_label, y_t))

"""# **Create Neural Network for Multiple Regression using Pytorch**"""

import torch
import torch.nn as nn
import torch.optim as optim

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

"""input features are 4, output 1"""

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
model

learningRate = 0.01
lossfunc = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

epochs_tensor = 1000

# Training
losses = []
for epoch in range(epochs_tensor):
    # Forward pass
    ypred_tensor = model(X_train_tensor)
    loss = lossfunc(ypred_tensor, y_train_tensor)
    losses.append(loss.item())

    # Backward pass and optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch % 50) == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

plt.figure(figsize=(12,6))
plt.plot(losses, 'm', lw=3)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.show()

"""Weight and Bias Values"""

print(model[0].weight.detach().numpy())
print(model[0].bias.detach().numpy())

"""Testing and evaluating the model"""

model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred_np = y_pred_tensor.numpy()
    y_test_np = y_test_tensor.numpy()

y_pred_tensor = y_pred_tensor.detach().numpy()
y_test_tensor = y_test_tensor.detach().numpy()

print('\nEvaluation Metrics:')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_np, y_pred_np)))
print('MAE:', metrics.mean_absolute_error(y_test_np, y_pred_np))
print('R2:', metrics.r2_score(y_test_np, y_pred_np))

"""Good perfomance, highly accurate. The low RMSE and MAE values further confirm that the model's predictions are very close to the actual values."""
