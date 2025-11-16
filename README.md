# Industrial Energy Consumption Forecasting

This project uses machine learning to forecast energy consumption in the steel industry based on historical data. The goal is to build a predictive model that can help in optimizing energy usage.

The core of the project includes data preprocessing, feature engineering, and training an XGBoost Regressor model. A simple web application is also provided to interact with the trained model.

## Features

* **Data Cleaning:** Processes raw steel industry data to handle missing values and prepare it for analysis.
* **Feature Engineering:** Creates new features to improve model performance.
* **Model Training:** Uses an `XGBoost` model to forecast `Usage_kWh`.
* **Web Application:** A simple front-end (built with Flask/Streamlit) to load the model and make live predictions.
* **Analysis:** A Jupyter Notebook (`Industrial_Energy_Consumption_Prediction.ipynb`) detailing the entire process.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn
* Flask / Streamlit (for `app.py`)
* Jupyter Notebook

## Project Structure

```
industrial-energy-consumption-forecasting/
├── app.py                      # Main application file (Flask/Streamlit)
├── data/
│   ├── processed/
│   │   └── cleaned_steel_data.csv  # Cleaned data
│   └── raw/
│       └── Steel_industry_data.csv # Raw dataset
├── notebooks/
│   └── Industrial_Energy_Consumption_Prediction.ipynb # Jupyter Notebook with full analysis
├── .gitignore                  # Files to ignore
├── README.md                   # You are here
├── requirements.txt            # Project dependencies
└── xgb_model.pkl               # Trained XGBoost model file
```

## How to Use

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

* Python 3.8 or higher
* Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/industrial-energy-consumption-forecasting.git](https://github.com/your-username/industrial-energy-consumption-forecasting.git)
    cd industrial-energy-consumption-forecasting
    ```

2.  **Create a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate
    
    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebook

To explore the data analysis, feature engineering, and model training process, you can run the Jupyter Notebook:

```bash
jupyter notebook notebooks/Industrial_Energy_Consumption_Prediction.ipynb
```

### Running the Application

To start the web application and make predictions:

**(Note: Change this command based on whether you used Flask or Streamlit)**

* **If using Flask:**
    ```bash
    python app.py
    ```
    Then open your browser to `http://127.0.0.1:5000`

* **If using Streamlit:**
    ```bash
    streamlit run app.py
    ```
    Then open your browser to the URL provided in the terminal.

## Data

The dataset used is `Steel_industry_data.csv`, which contains various features related to steel production and the corresponding energy consumption (`Usage_kWh`).

*(Optional: Add a link to where you got the dataset, e.g., "The dataset was sourced from Kaggle [link here].")*

## Model

The model is an `XGBoost Regressor` trained on the cleaned and feature-engineered data. It is saved in the file `xgb_model.pkl`. The notebook `Industrial_Energy_Consumption_Prediction.ipynb` contains the complete details of the model's training and evaluation.