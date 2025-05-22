# ALBASTRO - DEBUGGING = ALBASTRO_FIXED
# All codes are written by ALBASTRO, CANETE, YANEZ

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt # Matplotlib
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from streamlit_extras.metric_cards import style_metric_cards
from sklearn.model_selection import train_test_split    # Train-Test Split
from sklearn.linear_model import LinearRegression # Linear Regression
from sklearn.cluster import KMeans #K-means clustering
from sklearn.preprocessing import StandardScaler # StandardScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Page configuration
st.set_page_config(page_title="25-IS-004", page_icon="🌎", layout="wide")

# Load CSS style
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar logo
st.sidebar.image("images/logo2.png")

# Title
st.title("⏱ HAMILE GEN. MERCHANDISE ANALYTICS DASHBOARD")

# Load dataset
df = pd.read_excel("Hamile.xlsx")

# Date filter
start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365 * 4))
end_date = st.sidebar.date_input(label="End Date")

# Apply date filter
df_filtered = df[(df['Order Date'] >= str(start_date)) & (df['Order Date'] <= str(end_date))]

# Sidebar filter
st.sidebar.header("Please filter")
city = st.sidebar.multiselect("Select City", options=df_filtered["City"].unique(), default=df_filtered["City"].unique())
category = st.sidebar.multiselect("Select Category", options=df_filtered["Category"].unique(), default=df_filtered["Category"].unique())
region = st.sidebar.multiselect("Select Region", options=df_filtered["Region"].unique(), default=df_filtered["Region"].unique())

# Apply filters
df_selection = df_filtered.query("City == @city & Category == @category & Region == @region")

# Key performance metrics
st.subheader('Key Performance')

col1, col2, col3, col4 = st.columns(4)
col1.metric(label="⏱ Total Items", value=df_selection["Product"].count(), delta="Number of Items in stock")
col2.metric(label="⏱ Sum of Product Total Price PHP", value=f"{df_selection['Total Price (₱)'].sum():,.0f}", delta=df_selection['Total Price (₱)'].median())
col3.metric(label="⏱ Maximum Price PHP", value=f"{df_selection['Total Price (₱)'].max():,.0f}", delta="High Price")
col4.metric(label="⏱ Minimum Price PHP", value=f"{df_selection['Total Price (₱)'].min():,.0f}", delta="Low Price")
style_metric_cards(background_color="#00588E",border_left_color="#FF4B4B",border_color="#1f66bd",box_shadow="#F71938")

coll1, coll2 = st.columns(2)
coll1.info(f"Business Metrics between [{start_date}] and [{end_date}]")

# Create a Plotly bar chart for Product Quantity
fig_product = px.bar(
    df_selection,
    x="Product",
    y="Product Quantity",
    title="Product by Quantity",
    color="Category",
    labels={"Product Quantity": "Quantity"},
    text_auto=True
)

# Display the chart in Streamlit
st.plotly_chart(fig_product, use_container_width=True)

# Progress bar using Plotly Gauge Chart
def progress_bar():
    target = 500000
    current = df_selection["Total Price (₱)"].sum()
    percent = round((current / target * 100))

    fig_progress = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        title={"text": "Target Percentage"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "green"},
            "steps": [
                {"range": [0, 50], "color": "lightgray"},
                {"range": [50, 100], "color": "yellow"}
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 100}
        }
    ))

    st.plotly_chart(fig_progress, use_container_width=True)

with coll1:
    st.subheader("Target Percentage")
    progress_bar()

# Order date vs quantity bar chart using Plotly
with coll2:
    st.subheader("Product Order Date by Quantity")
    fig_order_date = px.bar(
        df_selection,
        x="Order Date",
        y="Product Quantity",
        title="Order Date vs Quantity",
        labels={"Product Quantity": "Quantity"},
        text_auto=True
    )

    st.plotly_chart(fig_order_date, use_container_width=True)
 #---------------------------------------------------EMERGING TECH STARTS HERE-----------------------------------------------------------------------

# ✅ Load dataset (Hamile.xlsx with "DataSet" sheet name)
df = pd.read_excel("Hamile.xlsx", sheet_name="DataSet")

# ✅ Remove extra spaces from column names
df.columns = df.columns.str.strip()

# ✅ Sidebar for user customization
st.sidebar.header("🔧 Prediction Controls For Sales Prediction")
model_choice = st.sidebar.selectbox("Choose Machine Learning Model", [
    "Linear Regression - Simple & Fast",
    "Decision Tree - Captures Complex Patterns",
    "Random Forest - Best Accuracy"
])

# ✅ Select features and target (Updated for Hamile.xlsx)
df_ml = df[["Product Quantity", "Total Price (₱)"]]  # Matches the new dataset
X = df_ml[["Product Quantity"]]  # Feature (Sales Volume)
y = df_ml["Total Price (₱)"]   # Target (Total Price)

# ✅ Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Initialize and Train Selected Model
if "Linear Regression" in model_choice:
    model = LinearRegression()
elif "Decision Tree" in model_choice:
    model = DecisionTreeRegressor()
elif "Random Forest" in model_choice:
    model = RandomForestRegressor(n_estimators=100)

model.fit(X_train, y_train)  # Train the model

# ✅ Predict Sales
y_pred = model.predict(X_test)

# ✅ Display Model Accuracy Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.title("📈 Sales Prediction using Machine Learning")
st.subheader("📊 Model Accuracy Metrics")
st.write(f"✅ Mean Absolute Error: **{mae:.2f}**")
st.write(f"✅ R-squared Score: **{r2:.2f}**")

# ✅ Predict User Input Quantity Dynamically
st.subheader("🔍 Predict Total Price Based on Product Quantity")
quantity_input = st.slider("Select Product Quantity to Predict Price", min_value=1, max_value=500, value=10)

if quantity_input:
    quantity_df = pd.DataFrame({"Product Quantity": [quantity_input]})  # Convert input into DataFrame
    predicted_price = model.predict(quantity_df)[0]
    st.success(f"**Predicted Total Price for {quantity_input} items:** ₱{predicted_price:,.2f}")

# ✅ Interactive Line Chart for Predictions
quantities = np.arange(1, 500, 5)  # Range of quantities
predicted_prices = model.predict(pd.DataFrame({"Product Quantity": quantities}))

fig = px.line(
    x=quantities, y=predicted_prices,
    labels={"x": "Product Quantity", "y": "Predicted Total Price"},
    title="📊 Interactive Sales Price Prediction"
)
st.plotly_chart(fig, use_container_width=True)  # ✅ Corrected Interactive Visualization
#--------------------------------------------------------------------------------------------------------------------------

# ✅ Load dataset (Using the uploaded file)
df = pd.read_excel("Hamile.xlsx")  # Updated dataset reference

# ✅ Remove extra spaces from column names
df.columns = df.columns.str.strip()

# ✅ Sidebar for user customization
st.sidebar.header("🔧 Clustering Controls for KMeans Clustering")
num_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)  # Dynamic cluster selection
selected_features = st.sidebar.multiselect(
    "Select Features for Clustering",
    ["Product Quantity", "Total Price (₱)", "Unit Price (₱)"],  # Updated column names
    default=["Product Quantity", "Total Price (₱)"]
)  # User-selected clustering features

# ✅ Prepare features for clustering
df_cluster = df[selected_features]

# ✅ Scale features (Standardizing ensures fair clustering)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

# ✅ Apply K-Means clustering with dynamic settings
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(df_scaled)

# ✅ Show clustered data
st.title("🔍 Interactive Product Clustering using K-Means")
st.subheader("📌 Clustered Products:")
styled_df = df[["Product", "Product Quantity", "Total Price (₱)", "Unit Price (₱)", "Cluster"]]  # Updated column names
st.dataframe(styled_df, height=600, width=900)  # Adjustable DataFrame display

# ✅ Interactive visualization using Plotly
st.subheader("🌍 Interactive Cluster Visualization")
fig = px.scatter(
    df,
    x="Product Quantity",  # Updated column reference
    y="Total Price (₱)",  # Updated column reference
    color="Cluster",
    hover_data=["Product"],
    title="Dynamic K-Means Clustering",
    labels={"Product Quantity": "Quantity Sold", "Total Price (₱)": "Total Price"},
    width=900,
    height=600
)

st.plotly_chart(fig, use_container_width=True)  # Interactive visualization with hover tool

#--------------------------------------------------------------------------------------------------------------------------

# Page title
st.title("📊 Interactive Time Series Forecasting using ARIMA")

# ✅ Load dataset
df = pd.read_excel("Hamile.xlsx")

# ✅ Convert 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# ✅ Sidebar: Region filter
st.sidebar.header("🔧 Time Series Controls")
selected_region = st.sidebar.selectbox("Select Region", df["Region"].unique())
filtered_data = df[df["Region"] == selected_region]

# ✅ Aggregate sales by date
sales_data = filtered_data.groupby("Order Date")["Total Price (₱)"].sum().reset_index()
sales_data.set_index("Order Date", inplace=True)

# ✅ Show data preview
st.subheader(f"🔍 Sales Data for {selected_region}")
st.write(sales_data.tail())

# ✅ Sidebar: Forecast duration
forecast_steps = st.sidebar.slider("Forecast Duration (Days)", 30, 365, 180)

# ✅ Sidebar: ARIMA parameters
p = st.sidebar.slider("AR (p)", 0, 10, 5)
d = st.sidebar.slider("Differencing (d)", 0, 2, 1)
q = st.sidebar.slider("MA (q)", 0, 10, 0)
use_sarimax = st.sidebar.checkbox("Use SARIMAX (Seasonality)", value=True)

# ✅ Function: ADF stationarity check
def check_stationarity(data):
    result = adfuller(data)
    st.write(f"ADF Statistic: {result[0]:.2f}")
    st.write(f"p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        st.success("✅ Series is stationary.")
    else:
        st.warning("⚠️ Series is NOT stationary. Consider differencing.")

# ✅ Initial stationarity check
st.subheader("📏 Stationarity Check")
check_stationarity(sales_data["Total Price (₱)"])

# ✅ Differencing
sales_data["Sales_Diff"] = sales_data["Total Price (₱)"].diff()
sales_data.dropna(inplace=True)

# ✅ Recheck stationarity
check_stationarity(sales_data["Sales_Diff"])

# ✅ Plot ACF and PACF
st.subheader("📈 ACF and PACF Plots")
fig_acf, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(sales_data["Sales_Diff"], ax=axes[0])
axes[0].set_title("ACF")
plot_pacf(sales_data["Sales_Diff"], ax=axes[1])
axes[1].set_title("PACF")
st.pyplot(fig_acf)

# ✅ Fit ARIMA or SARIMAX
target_series = filtered_data.groupby("Order Date")["Total Price (₱)"].sum()
target_series = target_series.asfreq('D').fillna(0)  # Daily frequency and fill missing
if use_sarimax:
    model = SARIMAX(target_series, order=(p, d, q), seasonal_order=(1, 1, 1, 12))
else:
    model = ARIMA(target_series, order=(p, d, q))

model_fit = model.fit()

# ✅ Forecast
forecast = model_fit.forecast(steps=forecast_steps)
start_date = target_series.index.max() + pd.Timedelta(days=1)
future_dates = pd.date_range(start=start_date, periods=forecast_steps, freq='D')
forecast_series = pd.Series(forecast, index=future_dates)

# ✅ Debug info
st.subheader("📅 Forecast Date Info")
st.write("🗓️ Last date in dataset:", target_series.index.max().date())
st.write("🔮 First forecast date:", future_dates[0].date())
st.write("🔮 Last forecast date:", future_dates[-1].date())

# ✅ Plot historical + forecast
st.subheader("📊 Forecast Plot")
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=target_series.index,
    y=target_series.values,
    mode='lines',
    name='📈 Historical Sales',
    line=dict(color='blue')
))

# Forecast
fig.add_trace(go.Scatter(
    x=forecast_series.index,
    y=forecast_series.values,
    mode='lines',
    name='🔮 Forecasted Sales',
    line=dict(color='orange', dash='dash')
))

fig.update_layout(
    title=f"Forecast for {selected_region}",
    xaxis_title="Date",
    yaxis_title="Sales (₱)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ✅ Performance Metrics
st.subheader("📉 Model Performance")
mae = mean_absolute_error(target_series[d:], model_fit.fittedvalues[d:])  # align with differenced values
rmse = mean_squared_error(target_series[d:], model_fit.fittedvalues[d:], squared=False)
st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
