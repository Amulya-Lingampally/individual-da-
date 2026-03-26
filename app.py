
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="SpendWise AI", layout="wide")

# Sidebar
st.sidebar.title("💡 SpendWise AI")
st.sidebar.markdown("### Data-Driven Decision Engine")

menu = st.sidebar.radio("Navigate", [
    "📊 Executive Summary",
    "📈 Descriptive Analytics",
    "👥 Customer Segmentation",
    "🔗 Association Rules",
    "🤖 Predictive Models",
    "🎯 Prescriptive Actions",
    "🔮 New Customer Scorer"
])

# Upload
file = st.file_uploader("Upload SpendWise Dataset", type=["csv","xlsx"])

if file:
    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
else:
    st.warning("⚠ Please upload your SpendWise dataset")
    st.stop()

# Preprocessing
df_model = df.copy()
le = LabelEncoder()

for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop("App_Interest", axis=1)
y = df_model["App_Interest"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- EXECUTIVE ----------------
if menu == "📊 Executive Summary":
    st.title("📊 Executive Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Income", int(df["Income"].mean()))
    col2.metric("Avg Expenses", int(df["Expenses"].mean()))
    col3.metric("Avg Savings %", int(df["Savings"].mean()))

    fig = px.pie(df, names="Impulse_Buying", title="Impulse Behavior Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.info("Users with high impulse buying are key targets for SpendWise AI.")

# ---------------- DESCRIPTIVE ----------------
elif menu == "📈 Descriptive Analytics":
    st.title("📈 Descriptive Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.histogram(df, x="Income", color="Financial_Stress", title="Income Distribution"))

    with col2:
        st.plotly_chart(px.histogram(df, x="Expenses", color="Impulse_Buying", title="Expense Distribution"))

    st.plotly_chart(px.box(df, x="Impulse_Buying", y="Expenses", title="Spending vs Behavior"))

    st.success("High impulse users tend to spend more.")

# ---------------- SEGMENTATION ----------------
elif menu == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation")

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    st.plotly_chart(px.scatter(df, x="Income", y="Expenses",
                               color=df["Cluster"].astype(str),
                               title="Customer Segments"))

    st.info("Cluster groups show spending behavior differences.")

# ---------------- ASSOCIATION ----------------
elif menu == "🔗 Association Rules":
    st.title("🔗 Association Rules")

    basket = pd.get_dummies(df.astype(str))
    freq = apriori(basket, min_support=0.1, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1)

    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])

    st.success("Patterns help identify cross-spending behavior.")

# ---------------- PREDICTIVE ----------------
elif menu == "🤖 Predictive Models":
    st.title("🤖 Predictive Models")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
    st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
    st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

    feat = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    st.plotly_chart(px.bar(feat.sort_values(by="Importance", ascending=False),
                           x="Feature", y="Importance"))

# ---------------- PRESCRIPTIVE ----------------
elif menu == "🎯 Prescriptive Actions":
    st.title("🎯 Prescriptive Actions")

    st.success("Target users with high expenses & low savings.")
    st.info("Send alerts to high impulse users.")
    st.warning("Encourage budgeting for high-stress users.")

# ---------------- SCORER ----------------
elif menu == "🔮 New Customer Scorer":
    st.title("🔮 New Customer Prediction")

    age = st.number_input("Age", 18, 60)
    income = st.number_input("Income")
    expenses = st.number_input("Expenses")
    savings = st.number_input("Savings")

    if st.button("Predict"):
        model = RandomForestClassifier()
        model.fit(X_scaled, y)

        input_data = [[age, income, expenses, savings] + [0]*(X.shape[1]-4)]
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)

        st.success(f"Prediction: {pred[0]}")
