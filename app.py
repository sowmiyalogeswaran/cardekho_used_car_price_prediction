import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from streamlit_extras.colored_header import colored_header
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_option_menu import option_menu


# Home Page
def home_page():
    colored_header(
        label='Welcome to :blue[Home] Page 👋🏼',
        color_name='blue-70',
        description='Welcome to CarDekho Used Car Price Prediction!'
    )
    st.markdown("## :blue[*Project title*:]")
    st.subheader("*CarDekho Used Car Price Prediction*")
    st.markdown("## :blue[*Skills take away From This Project*:]")
    st.subheader("*Python scripting, Data Wrangling, EDA, Machine Learning, Streamlit.*")
    st.markdown("## :blue[*Domain*:]")
    st.subheader("*Automobile*")
    st.markdown("## :blue[*Problem Statement*:]")
    st.subheader("*The project aims to predict used car prices based on features such as car model, mileage, engine CC, and more.*")
    st.markdown("## :blue[*Results*:]")
    st.subheader("*The result will empower users to make informed decisions while buying or selling used cars.*")
    
    
# Data Filtering Page
def data_filtering_page():
    colored_header(
        label='You are in Data :blue[Filtering] page',
        color_name='blue-70',
        description='Filter the dataset based on specific columns.'
    )
    
    @st.cache_data
    def load_data():
        return pd.read_csv('Cleaned_Car_Dheko.csv')

    df = load_data()

    choice = st.sidebar.selectbox('Select a column to know unique values:', ['Car_Model', 'Manufactured_By'])
    
    if choice == 'Car_Model':
        unique_values = df['Car_Model'].unique()
        st.sidebar.dataframe(pd.DataFrame({'Car Models': unique_values}))
    else:
        unique_values = df['Manufactured_By'].unique()
        st.sidebar.dataframe(pd.DataFrame({'Car Brands': unique_values}))
    
    filter = dataframe_explorer(df)
    if st.button('Submit'):
        st.dataframe(filter)


# Data Analysis Page
def data_analysis_page():
    colored_header(
        label='You are in Data :blue[Analysis] page',
        color_name='blue-70',
        description='Analyze the dataset with visualizations.'
    )

    @st.cache_data
    def load_data():
        return pd.read_csv('Cleaned_Car_Dheko.csv')

    df = load_data()

    choice = st.selectbox("**Select an option to Explore**", df.drop('Car_Price', axis=1).columns)

    st.markdown(f"## :blue[Car Price vs {choice}]")
    fig = px.histogram(df, x=choice, y='Car_Price', width=950, height=500)
    st.plotly_chart(fig)

    st.markdown(f"## :blue[Which Car Brand is most available]")
    brand = df.groupby('Manufactured_By').size().reset_index(name='Count')
    bar = px.bar(brand, x='Manufactured_By', y='Count', width=950, height=500)
    st.plotly_chart(bar)


# Data Prediction Page
def data_prediction_page():
    st.title("Car Price Prediction")

    # Load data and model
    @st.cache_data
    def load_data():
        df = pd.read_csv('Cleaned_Car_Dheko.csv')
        return df

    df = load_data()

    # Encode categorical variables
    le_model = LabelEncoder()
    le_model.fit(df['Car_Model'])
    le_brand = LabelEncoder()
    le_brand.fit(df['Manufactured_By'])
    le_fuel = LabelEncoder()
    le_fuel.fit(df['Fuel_Type'])
    le_location = LabelEncoder()
    le_location.fit(df['City'])  

    def encode_categorical(value, encoder):
        try:
            return encoder.transform([value])[0]
        except ValueError:
            return -1  

    # Load trained model
    @st.cache_resource
    def load_model():
        with open('GradientBoost_model.pkl', 'rb') as file:
            return pickle.load(file)

    model = load_model()

    # Form inputs for prediction
    with st.form(key='prediction_form'):
        car_brand = st.selectbox("Select Car Brand", df['Manufactured_By'].unique())
        car_model = st.selectbox("Select Car Model", df[df['Manufactured_By'] == car_brand]['Car_Model'].unique())
        model_year = st.selectbox("Select Car Year", df['Car_Produced_Year'].unique())
        transmission = st.radio("Select Transmission", ["Automatic", "Manual"])
        location = st.selectbox("Select Location", df['City'].unique())  # Corrected column name
        km_driven = st.number_input("Enter Kilometers Driven", min_value=0)
        engine_cc = st.number_input("Enter Engine CC", min_value=0)
        mileage = st.number_input("Enter Mileage", min_value=0.0)
        fuel_type = st.selectbox("Select Fuel Type", df['Fuel_Type'].unique())
        no_of_owners = st.number_input("Enter Number of Owners", min_value=1)
        no_of_seats = st.number_input("Enter Number of Seats", min_value=2)

        submit_button = st.form_submit_button('Predict')

    if submit_button:
        if km_driven <= 0 or engine_cc <= 0 or mileage <= 0:
            st.error("Please enter valid positive values for KM Driven, Engine CC, and Mileage.")
        elif car_model == "" or car_brand == "":
            st.error("Invalid car model or brand selected.")
        else:
            car_model_encoded = encode_categorical(car_model, le_model)
            car_brand_encoded = encode_categorical(car_brand, le_brand)
            transmission_encoded = 0 if transmission == "Automatic" else 1
            location_encoded = encode_categorical(location, le_location)
            fuel_type_encoded = encode_categorical(fuel_type, le_fuel)

            input_data = [
                km_driven,
                transmission_encoded,
                car_model_encoded,
                model_year,
                engine_cc,
                mileage,
                location_encoded,
                fuel_type_encoded,
                no_of_owners,
                car_brand_encoded,
                no_of_seats
            ]

            # Make prediction
            prediction = model.predict([input_data])
            st.success(f"Predicted Car Price: ₹{prediction[0]:,.2f}")


# Main App
class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({'title': title, 'function': function})

    def run(self):
        selected_app = option_menu('Car Resale Price Prediction',
                                   ["Home", "Data Filtering", "Data Analysis", "Data Prediction"],
                                   icons=['house', 'search', 'reception-4', 'dice-5-fill'],
                                   menu_icon='cash-coin', default_index=0, orientation="vertical")
        for app in self.apps:
            if app['title'] == selected_app:
                app['function']()



# Main execution
if __name__ == "__main__":
    app = MultiApp()
    app.add_app("Home", home_page)
    app.add_app("Data Filtering", data_filtering_page)
    app.add_app("Data Analysis", data_analysis_page)
    app.add_app("Data Prediction", data_prediction_page)
    app.run() 


       

