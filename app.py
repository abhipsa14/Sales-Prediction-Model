import streamlit as st
import pandas as pd
import joblib 

st.title("Sales Prediction System")

st.write("This is a simple sales prediction system that uses a linear regression model to predict sales based on advertising budgets.")

model = joblib.load('model/model.lb')

uploaded_data = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_data is not None:
    df = pd.read_csv(uploaded_data)
    st.write("Uploaded Data:")
    st.dataframe(df)

    if st.button("Predict"):
        # Ensure the DataFrame has the same columns as the model expects
        df = pd.DataFrame(df, columns=['TV', 'Radio', 'Newspaper'])
        predictions = model.predict(df)
        st.write("Predictions:")
        st.dataframe(predictions)
        ## create a line chart
        chart_data=pd.DataFrame(
            df,columns=['TV','Radio','Newspaper']
            # random n distribution
        )
        st.line_chart(chart_data,x_label='Sources',y_label='Sales')