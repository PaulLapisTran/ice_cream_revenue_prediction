import streamlit as st
import pickle
st.title('Revenue Prediction')
input = st.number_input('Input temperature', -100, 50)
if st.button('Predict'):
    model = pickle.load(open('revenue_prediction.pickle', 'rb'))
    st.write('Revenue prediction:', model.predict(input))
