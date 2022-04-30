import pickle
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.base import BaseEstimator, TransformerMixin
from currency_converter import CurrencyConverter
from pathlib import Path
from xgboost.sklearn import XGBRegressor


folder_path = Path(__file__).parents[0]


class CarsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, col_name, thresh=10):
        self.values_dict = dict()
        self.thresh = thresh
        self.col_name = col_name

    def fit(self, X):
        for _, val in X[self.col_name].iteritems():
            if not val in self.values_dict.keys():
                self.values_dict[val] = 1
            else:
                self.values_dict[val] += 1
        self.most_popular_values = [val for val, count in self.values_dict.items() if count >= self.thresh]
        return self

    @staticmethod
    def check_value(value, most_popular_values):
        if pd.isna(value):
            return 'Unknown'
        else:
            return value if value in most_popular_values else 'Other'

    def transform(self, X):
        X[self.col_name] = X[self.col_name].apply(CarsTransformer.check_value,
                                                  args=(self.most_popular_values,))
        return X


class CarFeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.features_list = list()

    def fit(self, X, col_name='Features'):
        for _, features in X[col_name].iteritems():
            for feature in features:
                feature = feature.lower().replace(' ', '_').replace('-', '_')
                if not feature in self.features_list:
                    self.features_list.append(feature)
        return self

    @staticmethod
    def check_feature(x, new_feature):
        sample_features = [feature.lower().replace(' ', '_').replace('-', '_') for feature in x]
        return 1 if new_feature in sample_features else 0

    def transform(self, X, col_name='Features'):
        X_new = X.copy()
        for new_feature in self.features_list:
            X_new[new_feature] = X_new[col_name].apply(CarFeaturesTransformer.check_feature,
                                                       args=(new_feature,))
        return X_new.drop(columns=[col_name])


@st.cache(hash_funcs={XGBRegressor: id})
def load_model():
    with open(folder_path / 'web_app_data/simplified_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


@st.cache
def get_exchange_rate():
    converter = CurrencyConverter(fallback_on_missing_rate=True)
    USD_to_PLN = converter.convert(1, 'USD', 'PLN')
    return USD_to_PLN


@st.cache
def load_preprocessor():
    with open(folder_path / 'web_app_data/preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
    return preprocessor


@st.cache
def load_cols_info():
    with open(folder_path / 'web_app_data/Colour.pkl', 'rb') as file:
        colours = pickle.load(file)

    with open(folder_path / 'web_app_data/Condition.pkl', 'rb') as file:
        conditions = pickle.load(file)

    with open(folder_path / 'web_app_data/Drive.pkl', 'rb') as file:
        drives = pickle.load(file)

    with open(folder_path / 'web_app_data/Features.pkl', 'rb') as file:
        features = pickle.load(file)

    with open(folder_path / 'web_app_data/Fuel_type.pkl', 'rb') as file:
        fuel_types = pickle.load(file)

    with open(folder_path / 'web_app_data/Offer_location.pkl', 'rb') as file:
        offer_locations = pickle.load(file)

    with open(folder_path / 'web_app_data/Transmission.pkl', 'rb') as file:
        transmissions = pickle.load(file)

    with open(folder_path / 'web_app_data/Type.pkl', 'rb') as file:
        body_types = pickle.load(file)

    with open(folder_path / 'web_app_data/Vehicle_brand.pkl', 'rb') as file:
        brands = pickle.load(file)

    with open(folder_path / 'web_app_data/Vehicle_model.pkl', 'rb') as file:
        models = pickle.load(file)

    return colours, conditions, drives, features, fuel_types, offer_locations, transmissions, body_types, brands, models


@st.cache
def load_htmls():
    with open(folder_path / 'visualization/car_prices.html', 'r', encoding='utf-8') as file:
        prices_HTML = file.read()

    with open(folder_path / 'visualization/production_years.html', 'r', encoding='utf-8') as file:
        production_years_HTML = file.read()

    with open(folder_path / 'visualization/car_brands.html', 'r', encoding='utf-8') as file:
        brands_HTML = file.read()

    with open(folder_path / 'visualization/car_models.html', 'r', encoding='utf-8') as file:
        models_HTML = file.read()

    with open(folder_path / 'visualization/provinces.html', 'r', encoding='utf-8') as file:
        provinces_HTML = file.read()

    with open(folder_path / 'visualization/mileage.html', 'r', encoding='utf-8') as file:
        mileages_HTML = file.read()

    with open(folder_path / 'visualization/horsepowers.html', 'r', encoding='utf-8') as file:
        horsepowers_HTML = file.read()

    with open(folder_path / 'visualization/displacements.html', 'r', encoding='utf-8') as file:
        displacements_HTML = file.read()

    with open(folder_path / 'visualization/car_conditions.html', 'r', encoding='utf-8') as file:
        conditions_HTML = file.read()

    with open(folder_path / 'visualization/car_transmissions.html', 'r', encoding='utf-8') as file:
        transmissions_HTML = file.read()

    with open(folder_path / 'visualization/car_drives.html', 'r', encoding='utf-8') as file:
        drives_HTML = file.read()

    with open(folder_path / 'visualization/car_types.html', 'r', encoding='utf-8') as file:
        types_HTML = file.read()

    with open(folder_path / 'visualization/car_fuel_types.html', 'r', encoding='utf-8') as file:
        fuel_types_HTML = file.read()

    return prices_HTML, production_years_HTML, brands_HTML, models_HTML, provinces_HTML, mileages_HTML, \
           horsepowers_HTML, displacements_HTML, conditions_HTML, transmissions_HTML, drives_HTML, types_HTML, \
           fuel_types_HTML


def show_prediction_page():
    st.title('Car price estimation')
    st.write("""### Provide some informations to estimate the car price""")

    colours, conditions, drives, features, fuel_types, offer_locations, transmissions, body_types, brands, models = \
        load_cols_info()

    brand = st.selectbox('BRAND:', brands)
    model = st.selectbox('MODEL:', models[brand])
    condition = st.selectbox('CONDITION:', conditions)
    production_year = st.slider('PRODUCTION YEAR', min_value=1950, max_value=2022, step=1, value=2010)
    mileage_km = st.slider('MILEAGE (KM)', min_value=0, max_value=1000000, step=500, value=100000)
    fuel_type = st.selectbox('FUEL TYPE:', fuel_types)
    displacement_l = st.slider('ENGINE DISPLACEMENT (LITRES)', min_value=0.4, max_value=8.4, step=0.1, value=1.8)
    displacement_cm3 = displacement_l * 1000
    power_hp = st.slider('POWER (HP)', min_value=1, max_value=1400, step=1, value=100)
    transmission = st.selectbox('TRANSMISSION:', transmissions)
    drive = st.selectbox('DRIVE:', drives)
    body_type = st.selectbox('BODY TYPE:', body_types)
    colour = st.selectbox('COLOUR:', colours)
    offer_location = st.selectbox('LOCATION:', offer_locations)
    doors_number = st.slider('DOORS NUMBER', min_value=1, max_value=8, step=1, value=4)
    additional_features = st.multiselect('ADDITIONAL CAR FEATURES', features)

    estimate = st.button('ESTIMATE CAR PRICE')
    if estimate:
        regressor = load_model()
        preprocessor = load_preprocessor()

        X = pd.DataFrame({'Condition': [condition], 'Vehicle_brand': [brand], 'Vehicle_model': [model],
                          'Production_year': [production_year], 'Mileage_km': [mileage_km], 'Power_HP': [power_hp],
                          'Displacement_cm3': [displacement_cm3], 'Fuel_type': [fuel_type], 'Drive': [drive],
                          'Transmission': [transmission], 'Type': [body_type], 'Doors_number': [doors_number],
                          'Colour': [colour], 'Offer_location': [offer_location], 'Features': [additional_features]})
        X_prepared = preprocessor.transform(X)
        price_USD = regressor.predict(X_prepared)[0]
        USD_to_PLN = get_exchange_rate()
        price_PLN = USD_to_PLN * price_USD

        st.subheader(f'Estimated price: {price_PLN:,.0f} PLN (${price_USD:,.0f})')


def show_exploration_page():
    prices_HTML, production_years_HTML, brands_HTML, models_HTML, provinces_HTML, mileages_HTML, horsepowers_HTML, \
    displacements_HTML, conditions_HTML, transmissions_HTML, drives_HTML, types_HTML, fuel_types_HTML = load_htmls()

    st.title('Polish car market insight')
    st.write('(Based on 208,304 adverts posted from 2021-03-26 to 2021-05-05 '
             'on the popular polish car advertising website)')

    st.subheader('Cars prices distribution')
    components.html(prices_HTML, height=500)

    st.subheader('Production years distribution')
    components.html(production_years_HTML, height=500)

    st.subheader('The most popular car brands')
    components.html(brands_HTML, height=500)

    st.subheader('The most popular car models')
    components.html(models_HTML, height=500)

    st.subheader('Provinces with the most offers')
    components.html(provinces_HTML, height=500)

    st.subheader('Cars mileages distribution')
    components.html(mileages_HTML, height=500)

    st.subheader('Cars engine horsepowers distribution')
    components.html(horsepowers_HTML, height=500)

    st.subheader('Cars engine displacements distribution')
    components.html(displacements_HTML, height=500)

    components.html(conditions_HTML, height=450)
    components.html(transmissions_HTML, height=450)
    components.html(drives_HTML, height=450)
    components.html(types_HTML, height=450)
    components.html(fuel_types_HTML, height=450)


st.sidebar.write("""# What would you like to do?""")
page = st.sidebar.selectbox('Predict car price or explore car market in Poland', ('Predict', 'Explore'))

if page == 'Predict':
    show_prediction_page()
else:
    show_exploration_page()
