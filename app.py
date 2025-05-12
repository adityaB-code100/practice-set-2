import streamlit as st
import pandas as pd
import joblib


model = joblib.load('grade.joblib')
label_encoders = joblib.load('grade_encoders.joblib')

# Streamlit App UI
st.title("📚 Grade Prediction App")
st.write("Enter the student details to predict the final grade.")

# User Inputs
major = st.selectbox("Major", label_encoders['Major'].classes_)
study_hours = st.number_input("Study Hours", min_value=0, max_value=24, value=10, step=1)
attendance = st.selectbox("Attendance", label_encoders['Attendance'].classes_)
assignment_score = st.number_input("Assignment Score", min_value=0, max_value=100, value=85, step=1)

# Prediction Button
if st.button("Predict Final Grade"):
    # Encode categorical inputs
    encoded_major = label_encoders['Major'].transform([major])[0]
    encoded_attendance = label_encoders['Attendance'].transform([attendance])[0]

    # Create input dataframe for prediction
    input_df = pd.DataFrame({
        'Major': [encoded_major],
        'Study Hours': [study_hours],
        'Attendance': [encoded_attendance],
        'Assignment Score': [assignment_score]
    })

    # Predict
    prediction = model.predict(input_df)
    st.success(f"📈 Predicted Final Grade: **{prediction[0]:.2f}**")
