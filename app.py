# --- Streamlit Dashboard Code (Save as app.py) ---

import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np

# --- Page Configuration ---
# This should be the very first Streamlit command
st.set_page_config(
    page_title="Energy Prediction Dashboard",
    page_icon="🏭",
    layout="wide"
)

# --- Caching ---
# Cache data loading to speed up app
@st.cache_data
def load_data(filepath='steel_data_with_features.csv'):
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Data file '{filepath}' not found. Please make sure it's in the same folder.")
        return pd.DataFrame() # Return empty dataframe

    # Create a datetime column for plotting
    try:
        # Assume a base year (e.g., 2023) since 'year' is missing
        temp_date = pd.to_datetime(data[['month', 'day', 'hour']].assign(year=2023))
        data['datetime'] = temp_date
    except Exception as e:
        st.error(f"Error creating datetime: {e}")
        data['datetime'] = pd.NaT
    return data.sort_values(by='datetime')

# Cache model loading
@st.cache_resource
def load_model(filepath='xgb_model.pkl'):
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file '{filepath}' not found. Please run the notebook to train and save the model.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Load Data and Model ---
data = load_data()
model = load_model()

# --- Dashboard Title ---
st.title('🏭 Industrial Energy Consumption Dashboard')
st.markdown("An interactive dashboard to analyze and predict energy usage patterns.")

# --- Tabbed Interface ---
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "🔮 Live Prediction", "📈 Model Performance"])

# --- Tab 1: Exploratory Data Analysis ---
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    
    if not data.empty:
        # 1.1 Time Series Plot
        st.subheader("Energy Usage Over Time")
        st.write("Visualizing actual usage (blue) against the 3-hour rolling average (red).")
        fig_time = px.line(
            data, 
            x='datetime', 
            y=['Usage_kWh', 'Usage_kWh_rolling_avg_3hr'],
            title='Actual Usage vs. 3hr Rolling Average',
            labels={'value': 'Usage (kWh)', 'datetime': 'Date'},
            hover_data={'variable': True, 'value': ':.2f'}
        )
        # --- CHANGE 1 ---
        st.plotly_chart(fig_time, width='stretch')
        
        # 1.2 Heatmap
        st.subheader("Average Energy Usage Heatmap")
        st.write("This shows the average `Usage_kWh` for each hour of the day vs. the day of the week.")
        
        # Pivot data for heatmap
        heatmap_data = data.pivot_table(
            values='Usage_kWh', 
            index='hour', 
            columns='Day_of_week', 
            aggfunc='mean'
        )
        
        fig_heatmap = px.imshow(
            heatmap_data,
            labels=dict(x="Day of Week", y="Hour of Day", color="Avg. Usage (kWh)"),
            x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], # Assuming 0=Mon, 6=Sun
            title="Average Usage by Hour and Day"
        )
        # --- CHANGE 2 ---
        st.plotly_chart(fig_heatmap, width='stretch')
        
        # 1.3 Data Table
        if st.checkbox("Show complete feature-engineered data"):
            st.dataframe(data)
    else:
        st.warning("Data could not be loaded for EDA.")

# --- Tab 2: Live Prediction ---
with tab2:
    st.header("Live Energy Prediction 🔮")
    st.write("Adjust the features below to get a live prediction from the trained XGBoost model.")

    if model is None or data.empty:
        st.warning("Model or data not loaded. Please ensure 'xgb_model.pkl' and 'steel_data_with_features.csv' are in the same folder.")
    else:
        # Get feature names from the model
        model_features = model.get_booster().feature_names
        
        # Create input form
        with st.form("prediction_form"):
            st.subheader("Input Features")
            
            # Use columns for a cleaner layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                hour = st.slider("Hour of Day (0-23)", 0, 23, int(data['hour'].mean()))
                day_of_week = st.slider("Day of Week (0-6)", 0, 6, int(data['Day_of_week'].mean()))
                month = st.select_slider("Month (1-12)", options=list(range(1, 13)), value=int(data['month'].mean()))
                day = st.select_slider("Day (1-31)", options=list(range(1, 32)), value=int(data['day'].mean()))

            with col2:
                # Get the mean of the most important features for default values
                lag_usage = st.number_input("Previous Hour Usage (Usage_kWh_lag_1hr)", value=data['Usage_kWh_lag_1hr'].mean())
                rolling_avg = st.number_input("3hr Rolling Avg (Usage_kWh_rolling_avg_3hr)", value=data['Usage_kWh_rolling_avg_3hr'].mean())
                co2 = st.number_input("CO2 (tCO2)", value=data['CO2(tCO2)'].mean())
                lag_reactive_power = st.number_input("Lagging Reactive Power (kVarh)", value=data['Lagging_Current_Reactive.Power_kVarh'].mean())

            with col3:
                load_type = st.selectbox("Load Type", options=data['Load_Type'].unique(), index=0)
                week_status = st.selectbox("Week Status (0=Weekday, 1=Weekend)", options=data['WeekStatus'].unique(), index=0)
                lag_power_factor = st.number_input("Lagging Power Factor", value=data['Lagging_Current_Power_Factor'].mean())
                lead_power_factor = st.number_input("Leading Power Factor", value=data['Leading_Current_Power_Factor'].mean())
            
            # Submit button
            submitted = st.form_submit_button("Predict")

            if submitted:
                # Create a DataFrame from the inputs
                input_data = pd.DataFrame(columns=model_features)
                input_data.loc[0] = 0 # Initialize with zeros
                
                # Fill with user inputs
                input_data.loc[0, 'hour'] = hour
                input_data.loc[0, 'Day_of_week'] = day_of_week
                input_data.loc[0, 'month'] = month
                input_data.loc[0, 'day'] = day
                input_data.loc[0, 'Usage_kWh_lag_1hr'] = lag_usage
                input_data.loc[0, 'Usage_kWh_rolling_avg_3hr'] = rolling_avg
                input_data.loc[0, 'CO2(tCO2)'] = co2
                input_data.loc[0, 'Lagging_Current_Reactive.Power_kVarh'] = lag_reactive_power
                input_data.loc[0, 'Load_Type'] = load_type
                input_data.loc[0, 'WeekStatus'] = week_status
                input_data.loc[0, 'Lagging_Current_Power_Factor'] = lag_power_factor
                input_data.loc[0, 'Leading_Current_Power_Factor'] = lead_power_factor
                
                # Fill remaining features with their mean (a reasonable default)
                for col in model_features:
                    if col not in input_data:
                        if col in data.columns:
                            input_data.loc[0, col] = data[col].mean()
                        else:
                            input_data.loc[0, col] = 0 
                
                for col in model_features:
                    if col not in input_data:
                        input_data[col] = 0

                # Make prediction
                prediction = model.predict(input_data[model_features])
                
                # Display the prediction
                st.metric(label="Predicted Energy Usage (kWh)", value=f"{prediction[0]:.2f} kWh")
                
                with st.expander("Show input data used for prediction"):
                    st.dataframe(input_data[model_features])

# --- Tab 3: Model Performance ---
with tab3:
    st.header("Model Performance Metrics")
    st.write("These metrics are calculated on the 20% test set (unseen data) in the Jupyter Notebook.")
    
    st.subheader("Model Feature Importance (from XGBoost)")
    st.write("This plot shows which features the model found most useful for making predictions.")

    if model:
        try:
            # Get importance and feature names
            importance = model.get_booster().get_score(importance_type='gain')
            importance_df = pd.DataFrame({
                'Feature': importance.keys(),
                'Importance (Gain)': importance.values()
            }).sort_values(by='Importance (Gain)', ascending=False)
            
            # Create a bar chart
            fig_imp = px.bar(
                importance_df.head(20), 
                x='Importance (Gain)', 
                y='Feature', 
                orientation='h', 
                title='Top 20 Most Important Features'
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed")) # Show top feature at the top
            
            # --- CHANGE 3 ---
            st.plotly_chart(fig_imp, width='stretch')
            
        except Exception as e:
            st.error(f"Could not plot feature importance: {e}")
    else:
        st.warning("Model not loaded. Cannot display feature importance.")