# E-commerce Customer Spending Prediction

This repository contains a deep learning project focused on predicting the "Yearly Amount Spent" by e-commerce customers. The project utilizes both TensorFlow and PyTorch frameworks to build and evaluate multiple regression neural networks.

## Project Overview

The goal of this project is to analyze customer data from an e-commerce platform and predict their annual spending. This can be valuable for businesses to understand customer behavior, identify high-value customers, and tailor marketing strategies.

## Dataset

The dataset used in this project is "Ecommerce Customers_Kaggle.csv". It contains information about e-commerce customers, including:

* `Email`
* `Address`
* `Avatar`
* `Avg. Session Length`: Average session length in minutes.
* `Time on App`: Time spent on the mobile app in minutes.
* `Time on Website`: Time spent on the website in minutes.
* `Length of Membership`: How many years the customer has been a member.
* `Yearly Amount Spent`: The target variable, representing the yearly amount spent by the customer.

## Analysis and Feature Selection

The initial analysis involved exploring the dataset's structure, checking for missing values, and visualizing correlations between features and the target variable.

* **Data Cleaning:** Categorical features like 'Email', 'Address', and 'Avatar' were dropped as they are not directly numerical and were not considered relevant for this regression task.
* **Correlation Analysis:** A heatmap was used to visualize the correlation matrix of numerical features.
* **Top Feature Selection:** The top 4 features most correlated with 'Yearly Amount Spent' were identified: 'Length of Membership', 'Time on App', 'Avg. Session Length', and 'Time on Website'. These features were then used for training the models.
* **Feature Inspection:** Individual plots were created to inspect the relationship between each of the selected features and 'Yearly Amount Spent'. 'Length of Membership' visually appeared to be the most strongly correlated feature.

## Model Development

Both TensorFlow and PyTorch were used to build and train multiple regression neural networks.

* **Best Architecture:** The architecture `[16, 16, 16, 16]` (four hidden layers, each with 16 neurons) was identified as the best performing based on evaluation metrics.
* **Model Compilation:** The model was compiled with the `adam` optimizer and `mean squared error (mse)` as the loss function.
* **Training:** The model was trained for 400 epochs.
* **Loss Visualization:** A plot of the training loss over epochs is provided to show the model's convergence.
* **Weight and Bias Inspection:** The weights and biases of the first layer are printed.

### PyTorch Model

* **Data Conversion:** The preprocessed NumPy arrays were converted to PyTorch tensors.
* **Model Definition:** A sequential neural network model was defined with the same best architecture found in TensorFlow (`[16, 16, 16, 16]`). `ReLU` activation functions were used for hidden layers and a linear output layer for regression.
* **Loss Function and Optimizer:** `MSELoss` was used as the loss function and `Adam` optimizer with a learning rate of 0.01.
* **Training:** The model was trained for 1000 epochs.
* **Loss Visualization:** A plot of the training loss over epochs is provided.
* **Weight and Bias Inspection:** The weights and biases of the first layer are printed.

## Evaluation Metrics

The following metrics were used to evaluate the performance of both TensorFlow and PyTorch models:

* **Root Mean Squared Error (RMSE):** Measures the average magnitude of the errors. Lower values indicate better performance.
* **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual values. Lower values indicate better performance.
* **R-squared (R2) Score:** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher R2 score (closer to 1) indicates a better fit.

## Results and Conclusion

Both the TensorFlow and PyTorch models demonstrated good performance in predicting 'Yearly Amount Spent'. The low RMSE and MAE values, coupled with high R2 scores, indicate that the models' predictions are very close to the actual values. This suggests that the chosen features and neural network architectures are effective for this regression task.

