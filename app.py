import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title
st.title("AI Propensity Model for Farmers Adoption of Technology")

# Dataset
data = {
'Age':[25,40,35,50,30,28,45,38,32,47,29,41,36,52,33,27,44,39,31,48],
'Education':[3,1,2,1,3,2,1,2,3,1,2,3,2,1,3,2,1,2,3,1],
'Farm_Size':[2,1,3,2,3,2,1,3,2,1,3,2,3,1,2,3,1,2,3,1],
'Income':[50000,20000,40000,25000,45000,35000,22000,42000,38000,26000,41000,48000,37000,21000,46000,39000,23000,44000,47000,24000],
'Internet':[1,0,1,0,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,0],
'Adopt_Tech':[1,0,1,0,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,0]
}

df = pd.DataFrame(data)

# Show dataset
st.subheader("Farmer Dataset")
st.dataframe(df)

# Adoption graph
st.subheader("Technology Adoption Distribution")

fig1, ax1 = plt.subplots()
df['Adopt_Tech'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_xlabel("Adoption")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# Features and target
X = df[['Age','Education','Farm_Size','Income','Internet']]
y = df['Adopt_Tech']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

st.subheader("Model Performance")
st.metric("Model Accuracy", round(accuracy*100,2))

# Feature graph
st.subheader("Average Feature Values")

fig2, ax2 = plt.subplots()
X.mean().plot(kind='bar', ax=ax2)
st.pyplot(fig2)

# User Input
st.subheader("Enter Farmer Details")

age = st.slider("Age",20,60)
education = st.selectbox("Education Level",[1,2,3])
farm_size = st.slider("Farm Size",1,5)
income = st.number_input("Income",value=30000)
internet = st.selectbox("Internet Access",[0,1])

# Prediction
if st.button("Predict"):

    prediction = model.predict([[age,education,farm_size,income,internet]])
    probability = model.predict_proba([[age,education,farm_size,income,internet]])

    if prediction[0]==1:
        st.success("Farmer is Likely to Adopt Technology")
    else:
        st.error("Farmer is Unlikely to Adopt Technology")

    # Probability graph
    st.subheader("Adoption Probability")

    prob_df = pd.DataFrame({
        "Outcome":["Not Adopt","Adopt"],
        "Probability":[probability[0][0],probability[0][1]]
    })

    st.bar_chart(prob_df.set_index("Outcome"))