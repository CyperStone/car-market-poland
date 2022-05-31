import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from currency_converter import CurrencyConverter
from PyQt6.QtWidgets import QApplication, QWidget, QComboBox, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt


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


class MultiComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self._changed = False
        self.view().pressed.connect(self.handleItemPressed)

    def setItemChecked(self, index, checked=False):
        item = self.model().item(index, self.modelColumn())

        if checked:
            item.setCheckState(Qt.CheckState.Checked)
        else:
            item.setCheckState(Qt.CheckState.Unchecked)

    def handleItemPressed(self, index):
        item = self.model().itemFromIndex(index)

        if item.checkState() == Qt.CheckState.Checked:
            item.setCheckState(Qt.CheckState.Unchecked)
        else:
            item.setCheckState(Qt.CheckState.Checked)
        self._changed = True

    def hidePopup(self):
        if not self._changed:
            super().hidePopup()
        self._changed = False

    def itemChecked(self, index):
        item = self.model().item(index, self.modelColumn())
        return item.checkState() == Qt.CheckState.Checked


class MyWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.features_input = None
        self.setGeometry(500, 50, 400, 500)
        self.setWindowTitle("Car price estimator")
        self.setFixedWidth(600)
        self.setFixedHeight(750)
        self.folder_path = '.\\web_app_data'
        converter = CurrencyConverter(fallback_on_missing_rate=True)
        self.USD_to_PLN = converter.convert(1, 'USD', 'PLN')

        self.load_data()
        self.create_layout()

    def load_data(self):
        with open(os.path.join(self.folder_path, 'Colour.pkl'), 'rb') as file:
            self.colours = pickle.load(file)

        with open(os.path.join(self.folder_path, 'Condition.pkl'), 'rb') as file:
            self.conditions = pickle.load(file)

        with open(os.path.join(self.folder_path, 'Drive.pkl'), 'rb') as file:
            self.drives = pickle.load(file)

        with open(os.path.join(self.folder_path, 'Features.pkl'), 'rb') as file:
            self.features = pickle.load(file)

        with open(os.path.join(self.folder_path, 'Fuel_type.pkl'), 'rb') as file:
            self.fuel_types = pickle.load(file)

        with open(os.path.join(self.folder_path, 'Offer_location.pkl'), 'rb') as file:
            self.offer_locations = pickle.load(file)

        with open(os.path.join(self.folder_path, 'Transmission.pkl'), 'rb') as file:
            self.transmissions = pickle.load(file)

        with open(os.path.join(self.folder_path, 'Type.pkl'), 'rb') as file:
            self.body_types = pickle.load(file)

        with open(os.path.join(self.folder_path, 'Vehicle_brand.pkl'), 'rb') as file:
            self.brands = pickle.load(file)

        with open(os.path.join(self.folder_path, 'Vehicle_model.pkl'), 'rb') as file:
            self.models = pickle.load(file)

        with open(os.path.join(self.folder_path, 'preprocessor.pkl'), 'rb') as file:
            self.preprocessor = pickle.load(file)

        with open(os.path.join(self.folder_path, 'simplified_model.pkl'), 'rb') as file:
            self.regressor = pickle.load(file)

    def create_layout(self):
        layout = QVBoxLayout()

        info_label = QLabel('Provide below some information about the car')
        info_label.setFont(QFont("Sanserif", 12, QFont.Weight.ExtraBold))
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        brand_label = QLabel('BRAND:')
        brand_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        self.brand_input = QComboBox()
        self.brand_input.addItems(self.brands)
        self.brand_input.currentTextChanged.connect(self.brand_input_changed)

        model_label = QLabel('MODEL:')
        model_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        self.model_input = QComboBox()
        self.choosen_brand = self.brand_input.currentText()
        self.model_input.addItems(self.models[self.choosen_brand])

        condition_label = QLabel('CONDITION:')
        condition_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        self.condition_input = QComboBox()
        self.condition_input.addItems(self.conditions)

        production_year_label = QLabel('PRODUCTION YEAR:')
        production_year_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        production_year_hbox = QHBoxLayout()

        self.production_year_input = QSlider()
        self.production_year_input.setOrientation(Qt.Orientation.Horizontal)
        self.production_year_input.setTickPosition(QSlider.TickPosition.TicksAbove)
        self.production_year_input.setTickInterval(1)
        self.production_year_input.setMinimum(1950)
        self.production_year_input.setMaximum(2022)
        self.production_year_input.setValue(2010)
        self.production_year_input.valueChanged.connect(self.production_year_changed)

        self.production_year_result = QLabel('2010')
        self.production_year_result.setFont(QFont("Sanserif", 8, QFont.Weight.ExtraBold))

        production_year_hbox.addWidget(self.production_year_input)
        production_year_hbox.addSpacing(10)
        production_year_hbox.addWidget(self.production_year_result)

        mileage_label = QLabel('MILEAGE (KM):')
        mileage_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        mileage_hbox = QHBoxLayout()

        self.mileage_input = QSlider()
        self.mileage_input.setOrientation(Qt.Orientation.Horizontal)
        self.mileage_input.setTickPosition(QSlider.TickPosition.TicksAbove)
        self.mileage_input.setTickInterval(10)
        self.mileage_input.setMinimum(0)
        self.mileage_input.setMaximum(1000)
        self.mileage_input.setValue(100)
        self.mileage_input.valueChanged.connect(self.mileage_changed)

        self.mileage_result = QLabel('100000')
        self.mileage_result.setFont(QFont("Sanserif", 8, QFont.Weight.ExtraBold))

        mileage_hbox.addWidget(self.mileage_input)
        mileage_hbox.addSpacing(10)
        mileage_hbox.addWidget(self.mileage_result)

        fuel_type_label = QLabel('FUEL TYPE:')
        fuel_type_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        self.fuel_type_input = QComboBox()
        self.fuel_type_input.addItems(self.fuel_types)

        engine_displacement_label = QLabel('ENGINE DISPLACEMENT (LITRES):')
        engine_displacement_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        engine_displacement_hbox = QHBoxLayout()

        self.engine_displacement_input = QSlider()
        self.engine_displacement_input.setOrientation(Qt.Orientation.Horizontal)
        self.engine_displacement_input.setTickPosition(QSlider.TickPosition.TicksAbove)
        self.engine_displacement_input.setTickInterval(1)
        self.engine_displacement_input.setMinimum(4)
        self.engine_displacement_input.setMaximum(84)
        self.engine_displacement_input.setValue(18)
        self.engine_displacement_input.valueChanged.connect(self.engine_displacement_changed)

        self.engine_displacement_result = QLabel('1.8')
        self.engine_displacement_result.setFont(QFont("Sanserif", 8, QFont.Weight.ExtraBold))

        engine_displacement_hbox.addWidget(self.engine_displacement_input)
        engine_displacement_hbox.addSpacing(10)
        engine_displacement_hbox.addWidget(self.engine_displacement_result)

        power_label = QLabel('POWER (HP):')
        power_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        power_hbox = QHBoxLayout()

        self.power_input = QSlider()
        self.power_input.setOrientation(Qt.Orientation.Horizontal)
        self.power_input.setTickPosition(QSlider.TickPosition.TicksAbove)
        self.power_input.setTickInterval(50)
        self.power_input.setMinimum(1)
        self.power_input.setMaximum(1400)
        self.power_input.setValue(100)
        self.power_input.valueChanged.connect(self.power_changed)

        self.power_result = QLabel('100')
        self.power_result.setFont(QFont("Sanserif", 8, QFont.Weight.ExtraBold))

        power_hbox.addWidget(self.power_input)
        power_hbox.addSpacing(10)
        power_hbox.addWidget(self.power_result)

        transmission_label = QLabel('TRANSMISSION:')
        transmission_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        self.transmission_input = QComboBox()
        self.transmission_input.addItems(self.transmissions)

        drive_label = QLabel('DRIVE:')
        drive_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        self.drive_input = QComboBox()
        self.drive_input.addItems(self.drives)

        body_type_label = QLabel('BODY TYPE:')
        body_type_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        self.body_type_input = QComboBox()
        self.body_type_input.addItems(self.body_types)

        color_label = QLabel('COLOR:')
        color_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        self.color_input = QComboBox()
        self.color_input.addItems(self.colours)

        location_label = QLabel('LOCATION:')
        location_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        self.location_input = QComboBox()
        self.location_input.addItems(self.offer_locations)

        doors_label = QLabel('DOORS NUMBER:')
        doors_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        doors_hbox = QHBoxLayout()

        self.doors_input = QSlider()
        self.doors_input.setOrientation(Qt.Orientation.Horizontal)
        self.doors_input.setTickPosition(QSlider.TickPosition.TicksAbove)
        self.doors_input.setTickInterval(1)
        self.doors_input.setMinimum(1)
        self.doors_input.setMaximum(8)
        self.doors_input.setValue(4)
        self.doors_input.valueChanged.connect(self.doors_changed)

        self.doors_result = QLabel('4')
        self.doors_result.setFont(QFont("Sanserif", 8, QFont.Weight.ExtraBold))

        doors_hbox.addWidget(self.doors_input)
        doors_hbox.addSpacing(10)
        doors_hbox.addWidget(self.doors_result)

        features_label = QLabel('ADDITIONAL CAR FEATURES:')
        features_label.setFont(QFont("Sanserif", 6, QFont.Weight.ExtraBold))

        self.features_input = MultiComboBox()
        for i, feature in enumerate(self.features):
            self.features_input.addItem(feature)
            self.features_input.setItemChecked(i, False)

        self.result_button = QPushButton('ESTIMATE PRICE')
        self.result_button.setFont(QFont("Times", 8, QFont.Weight.ExtraBold))
        self.result_button.setFixedHeight(30)
        self.result_button.clicked.connect(self.estimate_price)

        self.result_label = QLabel('')
        self.result_label.setFont(QFont("Sanserif", 14, QFont.Weight.ExtraBold, italic=True))
        self.result_label.setStyleSheet('border: 1px solid black')
        self.result_label.setFixedHeight(35)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(info_label)
        layout.addWidget(brand_label)
        layout.addWidget(self.brand_input)
        layout.addWidget(model_label)
        layout.addWidget(self.model_input)
        layout.addWidget(condition_label)
        layout.addWidget(self.condition_input)
        layout.addWidget(production_year_label)
        layout.addLayout(production_year_hbox)
        layout.addWidget(mileage_label)
        layout.addLayout(mileage_hbox)
        layout.addWidget(fuel_type_label)
        layout.addWidget(self.fuel_type_input)
        layout.addWidget(engine_displacement_label)
        layout.addLayout(engine_displacement_hbox)
        layout.addWidget(power_label)
        layout.addLayout(power_hbox)
        layout.addWidget(transmission_label)
        layout.addWidget(self.transmission_input)
        layout.addWidget(drive_label)
        layout.addWidget(self.drive_input)
        layout.addWidget(body_type_label)
        layout.addWidget(self.body_type_input)
        layout.addWidget(color_label)
        layout.addWidget(self.color_input)
        layout.addWidget(location_label)
        layout.addWidget(self.location_input)
        layout.addWidget(doors_label)
        layout.addLayout(doors_hbox)
        layout.addWidget(features_label)
        layout.addWidget(self.features_input)
        layout.addWidget(self.result_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def brand_input_changed(self):
        self.choosen_brand = self.brand_input.currentText()
        self.model_input.clear()
        self.model_input.addItems(self.models[self.choosen_brand])

    def production_year_changed(self):
        value = self.production_year_input.value()
        self.production_year_result.setText(str(value))

    def mileage_changed(self):
        value = self.mileage_input.value()
        self.mileage_result.setText(str(value * 1000))

    def engine_displacement_changed(self):
        value = self.engine_displacement_input.value()
        self.engine_displacement_result.setText(str(np.round(value * 0.1, 1)))

    def power_changed(self):
        value = self.power_input.value()
        self.power_result.setText(str(value))

    def doors_changed(self):
        value = self.doors_input.value()
        self.doors_result.setText(str(value))

    def estimate_price(self):
        brand = self.brand_input.currentText()
        model = self.model_input.currentText()
        condition = self.condition_input.currentText()
        production_year = self.production_year_input.value()
        mileage = self.mileage_input.value() * 1000
        fuel_type = self.fuel_type_input.currentText()
        engine_displacement = self.engine_displacement_input.value() * 0.1 * 1000
        power = self.power_input.value()
        transmission = self.transmission_input.currentText()
        drive = self.drive_input.currentText()
        body_type = self.body_type_input.currentText()
        color = self.color_input.currentText()
        location = self.location_input.currentText()
        doors = self.doors_input.value()
        features = [self.features[i] for i in range(self.features_input.count()) if self.features_input.itemChecked(i)]

        X = pd.DataFrame({'Condition': [condition], 'Vehicle_brand': [brand], 'Vehicle_model': [model],
                          'Production_year': [production_year], 'Mileage_km': [mileage], 'Power_HP': [power],
                          'Displacement_cm3': [engine_displacement], 'Fuel_type': [fuel_type], 'Drive': [drive],
                          'Transmission': [transmission], 'Type': [body_type], 'Doors_number': [doors],
                          'Colour': [color], 'Offer_location': [location], 'Features': [features]})
        X_prepared = self.preprocessor.transform(X)
        price_USD = self.regressor.predict(X_prepared)[0]
        price_PLN = self.USD_to_PLN * price_USD

        self.result_label.setText(f'Estimated price: {price_PLN:,.0f} PLN (${price_USD:,.0f})')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()

    window.show()
    sys.exit(app.exec())
