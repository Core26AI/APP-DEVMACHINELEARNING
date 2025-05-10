import streamlit as st
import pandas as pd
import altair as alt
from datetime import date, timedelta
from streamlit_extras.metric_cards import style_metric_cards
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#page layout - YANEZ
st.set_page_config(page_title="25-IS-004", page_icon="🌎", layout="wide")

#streamlit theme=none - YANEZ
theme_plotly = None 

#sidebar logo - YANEZ
st.sidebar.image("images/logo2.png")

#title -  YANEZ
st.title("⏱ ER SUPERMARKET ANALYTICS DASHBOARD")

# load CSS Style - CANETE
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#load dataset - CANETE
df = pd.read_excel("foodsales.xlsx",sheet_name="FoodSales")


#date filter - ALBASTRO
start_date=st.sidebar.date_input("Start Date",date.today()-timedelta(days=365*4))
end_date=st.sidebar.date_input(label="End Date")

#compare date - ALBASTRO
df2 = df[(df['OrderDate'] >= str(start_date)) & (df['OrderDate'] <= str(end_date))]

#sidebar switcher  - CANETE
st.sidebar.header("Please filter")
city=st.sidebar.multiselect(
    "Select City",
     options=df2["City"].unique(),
     default=df2["City"].unique(),
)
category=st.sidebar.multiselect(
    "Select Category",
     options=df2["Product"].unique(),
     default=df2["Product"].unique(),
)
region=st.sidebar.multiselect(
    "Select Region",
     options=df2["Region"].unique(),
     default=df2["Region"].unique(),
)

df_selection=df2.query(
    "City==@city & Product==@category & Region ==@region"
)

#metrics - YANEZ
st.subheader('Key Performance')

col1, col2,col3,col4 = st.columns(4)
col1.metric(label="⏱ Total Items ", value=df_selection.Product.count(), delta="Number of Items in stock")
col2.metric(label="⏱ Sum of Product Total Price PHP:", value= f"{df_selection.TotalPrice.sum():,.0f}",delta=df_selection.TotalPrice.median())
col3.metric(label="⏱ Maximum Price  PHP:", value= f"{ df_selection.TotalPrice.max():,.0f}",delta="High Price")
col4.metric(label="⏱ Minimum Price  PHP:", value= f"{ df_selection.TotalPrice.min():,.0f}",delta="Low Price")
style_metric_cards(background_color="#00588E",border_left_color="#FF4B4B",border_color="#1f66bd",box_shadow="#F71938")
 

coll1,coll2=st.columns(2)
coll1.info("Business Metrics between[ "+str(start_date)+"] and ["+str(end_date)+"]")

#bar chart - ALBASTRO
with coll1:
 st.subheader("Product by Quantity")
 source = pd.DataFrame({
        "Quantity ($)": df_selection["Quantity"],
        "Product": df_selection["Product"]
      })
 
 bar_chart = alt.Chart(source).mark_bar().encode(
        x="sum(Quantity ($)):Q",
        y=alt.Y("Product:N", sort="-x")

    )
 st.altair_chart(bar_chart, use_container_width=True,theme=theme_plotly,)

#Progress Bar - ALBASTRO
def Progressbar():
    st.markdown("""<style>.stProgress > div > div > div > div { background-image: linear-gradient(to right, #99ff99 , #FFFF00)}</style>""",unsafe_allow_html=True,)
    target=50000
    current=df_selection["TotalPrice"].sum()
    percent=round((current/target*100))
    mybar=st.progress(0)
    if percent>100:
        st.subheader("Target done !")
    else:
     st.write("you have ",percent, "% " ,"of ", (format(target, 'd')), "PHP")
     mybar.progress(percent,text=" Target Percentage")
with coll1:
 st.subheader("Target Percentage")
 Progressbar()


#bar chart - ALBASTRO
with coll2:
 st.subheader("Product OrderDate by Quantity")
 data = {
    'Category': df_selection['OrderDate'],
    'Value': df_selection['Quantity'],
 }
 df = pd.DataFrame(data)
 st.bar_chart(df.set_index('Category')['Value'],use_container_width=True, width=600, height=600,)

# Machine Learning - Linear Regression -ALBASTRO
st.subheader("📈 Sales Prediction using Machine Learning")

# Select features and target - Yanez
df_ml = df_selection[["Quantity", "TotalPrice"]]
X = df_ml[["Quantity"]]  # Feature
y = df_ml["TotalPrice"]   # Target

# Split data into training and testing sets- YANEZ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model - YANEZ
model = LinearRegression()
model.fit(X_train, y_train)

# Predict - YANEZ
y_pred = model.predict(X_test)

# Display results - YANEZ
st.write("**Actual vs Predicted Total Price:**")
results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
st.dataframe(df_selection.style.set_properties(**{'width': '150px'}))

# Predict user input quantity - YANEZ, ALBASTRO (FIXED_WARNINGS)
quantity_input = st.number_input("Enter Quantity to Predict Price", min_value=1, step=1)

if quantity_input:
    # Convert input into DataFrame to match feature names - YANEZ, ALBASTRO (FIXED_WARNINGS)
    quantity_df = pd.DataFrame({"Quantity": [quantity_input]})  
    predicted_price = model.predict(quantity_df)[0]
    
    st.write(f"**Predicted Total Price for {quantity_input} items:** ${predicted_price:,.2f}")

#--------------------------------------------------------------------------------------------------------------------------

# --- 🔥 K-Means Clustering for Product Segmentation --- CANETE
st.subheader("🔍 Product Clustering using K-Means")

# Prepare features for clustering - CANETE
df_cluster = df_selection[["Quantity", "TotalPrice"]]

# Scale features (Standardizing ensures fair clustering) CANETE
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

# Apply K-Means clustering - CANETE
num_clusters = 3  # You can adjust this based on analysis - CANETE
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)  # ✅ ALBASTRO - Explicitly set n_init to suppress warning
df_selection["Cluster"] = kmeans.fit_predict(df_scaled)

# Show clustered data - CANETE
st.write("🔍 Clustered Products:")
styled_df = df_selection[["Product", "Quantity", "TotalPrice", "Cluster"]]

# ✅ Apply styling without breaking clustering graph - CANETE
st.dataframe(styled_df.style.set_properties(**{'width': '150px'}))

# Visualize Clusters with adjusted size - YANEZ
fig, ax = plt.subplots(figsize=(10, 6))  # ✅ Adjusted figure size - ALBASTRO
scatter = ax.scatter(df_selection["Quantity"], df_selection["TotalPrice"], c=df_selection["Cluster"], cmap="viridis", alpha=0.6)
ax.set_xlabel("Quantity Sold")
ax.set_ylabel("Total Price")
ax.set_title("K-Means Clustering of Products")

# Show plot in Streamlit - CANETE
st.pyplot(fig)