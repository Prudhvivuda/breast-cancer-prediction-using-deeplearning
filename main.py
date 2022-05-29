import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

pickle_in = open("classifier.pkl","rb")
ann_classifier = pickle.load(pickle_in)
pickle_in.close()

def welcome():
    return "welcome all"        

def predict(test):
    prediction = ann_classifier.predict(test)
    print(prediction)
    return prediction

def main(sc):
    st.title("Breast Cancer Predictor")
    html_temp = """
    <div style="background-color:green;padding:20px"\>
    <h2 style="color:white;text-align:center;">Breast Cancer Prediction APP</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    radius_mean = st.number_input("Radius Mean", step=1., format="%.5f", value=13.54000)
    texture_mean = st.number_input("Texture Mean",step=1., format="%.5f", value=21.60400)
    perimeter_mean	 = st.number_input("Perimeter Mean", step=1., format="%.5f", value=115.30000)
    area_mean = st.number_input("Area Mean", step=1., format="%.5f", value=978.37640)
    smoothness_mean = st.number_input("Smoothness Mean", step=1., format="%.5f", value=0.10289)
    compactness_mean = st.number_input("Compactness Mean", step=1., format="%.5f", value=0.14518)
    concavity_mean = st.number_input("Concavity Mean", step=1., format="%.5f", value=0.16077)
    concave_points_mean	 = st.number_input("Concave points", step=1., format="%.5f", value=0.08799)
    symmetry_mean = st.number_input("Symmetry Mean", step=1., format="%.5f", value=0.19290)
    radius_se = st.number_input("Radius_se", step=1., format="%.5f", value=0.60908)
    perimeter_se = st.number_input("Perimeter_se", step=1., format="%.5f", value=4.32392)
    area_se = st.number_input("Area_se", step=1., format="%.5f", value=72.67241)
    compactness_se = st.number_input("Compactness_se", step=1., format="%.5f", value=0.03228)
    concavity_se = st.number_input("Concavity_se", step=1., format="%.5f", value=0.04182)
    concave_points_se = st.number_input("Concave points_se", step=1., format="%.5f", value=0.01506)
    fractal_dimension_se = st.number_input("Fractal_dimension_se", step=1., format="%.5f", value=0.00406)
    radius_worst = st.number_input("Radius_worst", step=1., format="%.5f", value=21.13481)
    texture_worst = st.number_input("Texture_worst", step=1., format="%.5f", value=29.31821)
    perimeter_worst = st.number_input("Perimeter_worst", step=1., format="%.5f", value=141.37030)
    area_worst = st.number_input("Area_worst", step=1., format="%.5f", value=1422.28600)
    smoothness_worst = st.number_input("Smoothness_worst", step=1., format="%.5f", value=0.14222)
    compactness_worst = st.number_input("Compactness_worst", step=1., format="%.5f", value=0.37482)
    concavity_worst = st.number_input("Concavity_worst", step=1., format="%.5f", value=0.45060)
    concave_points_worst = st.number_input("Concave points_worst", step=1., format="%.5f", value=0.18224)
    symmetry_worst = st.number_input("Symmetry_worst", step=1., format="%.5f", value=0.32347)
    fractal_dimension_worst = st.number_input("Fractal_dimension_worst", step=1., format="%.5f", value=0.09153)
    
    if st.button("Predict"):
        
        # have breast cancer malignant
        # data = sc.transform([[radius_mean, texture_mean,perimeter_mean,area_mean,smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, radius_se,
        #                         perimeter_se, area_se, compactness_se, concavity_se, concave_points_se,fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
        #                         compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]])

        # no breast cancer begnin
        data = sc.transform([[13.54,14.36,87.46,566.3,0.09779, 0.08129,0.06664,0.04781,0.1885,0.2699,2.058,23.56,0.0146,0.02387,0.01315,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]]) 

        print(data)
        result = predict(data)

        if result < [[0.5]]:
            print("zero")
            st.success('No Breast Cancer !')

        if result >= [[0.5]]:
            print("May have breast Cancer")
            st.success('May have breast Cancer (:')

if __name__=='__main__':

    data = pd.read_csv('data.csv')
    # del data['Unnamed: 32']
    del data['symmetry_se']
    del data['texture_se']
    del data['fractal_dimension_mean']
    del data['smoothness_se']
    X = data.iloc[:, 2:].values
    y = data.iloc[:, 1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    sc = StandardScaler()
    sc.fit_transform(X_train)

    main(sc)