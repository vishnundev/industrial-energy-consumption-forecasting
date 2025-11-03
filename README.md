# 🏭 Industrial Energy Consumption Forecasting

### 📊 Overview
This project uses **machine learning** to predict industrial energy consumption based on operational and environmental factors.  
The dataset is sourced from the **UCI Machine Learning Repository (Steel Industry Energy Consumption)** and contains one year of smart factory energy data collected from South Korea.

### 🎯 Objective
- Clean and preprocess raw industrial energy data.
- Perform feature engineering (time, environment, operational metadata).
- Develop regression models to forecast energy consumption.
- Analyze efficiency patterns and identify potential energy-saving opportunities.

### 🧰 Tech Stack
- **Languages:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Tools:** Jupyter Notebook, GitHub

### 📁 Project Structure
```
industrial-energy-consumption-forecasting/
│
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned dataset
│
├── notebooks/
│   └── Industrial_Energy_Consumption_Prediction.ipynb
│
├── README.md
├── requirements.txt
└── LICENSE
```

### 🔍 Dataset
**Source:** [UCI Machine Learning Repository - Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)  
**License:** Open for academic and research use.

### 🧪 Steps Performed
1. Loaded and inspected raw dataset  
2. Cleaned missing values and handled outliers  
3. Encoded categorical variables  
4. Created time-based features  
5. Trained regression model (Linear Regression, Random Forest)  
6. Evaluated performance using MAE and R² Score  

### 📈 Results
- Achieved **XX% R² score** and **MAE = XX kWh** on test data.  
- Found significant correlation between temperature, humidity, and energy usage.

### 💡 Future Work
- Implement anomaly detection to identify inefficient operations.  
- Deploy a Streamlit web dashboard for real-time monitoring.  
- Integrate weather API data for live forecasting.

### 👨‍💻 Author
**Vishnu N**  
📧 vishnun2811@gmail.com

---

> ⭐ If you found this project useful, please consider starring the repo!
