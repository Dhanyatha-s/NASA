# NASA Turbofan Engine Degradation Prediction
## Project Overview
This project aims to predict the Remaining Useful Life (RUL) of aircraft engines using the NASA Turbofan Engine Degradation Simulation dataset. The predictive model developed in this project can assist in proactive maintenance, enhancing the safety and efficiency of aircraft operations.

|Table of Contents|  
___________________
|Background|  
|Data Description|  
|Project Structure|  
|Installation|  
|Usage|  
|Model Development|  
|Results|  
|Future Work|  
|License|  
|Acknowledgements|  

## Background
Aircraft engines undergo significant wear and tear over time. Predicting the RUL of an engine is crucial for maintenance scheduling and ensuring operational safety. This project leverages historical sensor data to forecast RUL, enabling better maintenance decision-making.

## Data Description
The dataset used in this project is the NASA Turbofan Engine Degradation Simulation dataset, which contains:

Multiple sensor readings (e.g., temperature, pressure) recorded during engine operation.
A target variable indicating the Remaining Useful Life (RUL) of the engine.
The data is organized into two primary files:

train_data.csv: Contains training data with sensor measurements and corresponding RUL values.
test_data.csv: Contains test data for evaluating the model's performance.
Project Structure
```
Copy code
├── data/
│   ├── train_data.csv
│   ├── test_data.csv
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_development.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
├── requirements.txt
├── README.md
```
### Installation
Clone the repository:

```bash
Copy code
git clone https://github.com/yourusername/turbofan-engine-degradation.git
cd turbofan-engine-degradation
Install the required packages:
```
```bash
Copy code
pip install -r requirements.txt
Usage
To run the analysis and model training:
```
Load the dataset:  nasa-turbofan-engine-degradation-simulation-data 


train_data = pd.read_csv('data/train_data.csv')
test_data = pd.read_csv('data/test_data.csv')
Run the Jupyter notebooks for exploratory analysis and model development:

## Model Development
Data Preprocessing: Cleaned and merged datasets, removed missing values, and scaled features.
Feature Selection: Identified key features influencing RUL using correlation analysis.
Model Selection: Implemented a Random Forest Regressor to predict RUL.
Model Evaluation: Evaluated model performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
## Results
The model achieved the following performance metrics on the test dataset:

Mean Squared Error (MSE): 1897.74
Mean Absolute Error (MAE): 37.54
Visualizations of the predicted vs. actual RUL values indicate a strong correlation, demonstrating the model's effectiveness.

## Future Work
Explore advanced modeling techniques such as Gradient Boosting or LSTM networks.
Conduct hyperparameter tuning to improve model performance.
Implement cross-validation for more robust evaluation.
Investigate anomaly detection techniques for early failure prediction.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
NASA for providing the dataset.
The open-source community for libraries and tools that made this project possible.
