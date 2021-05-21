import matplotlib.pyplot as plt
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
import geopandas as gpd


DATA_PATH = "data_final_8_cat.csv"
COLS_TO_REMOVE = ["Date", "date", "CountryName", "iso_code","continent", "location", "RegionName", "RegionCode"]
df = pd.read_csv(DATA_PATH)
df.drop(columns=COLS_TO_REMOVE, inplace=True)
df.dropna(inplace=True)
gb = df.groupby("CountryCode")
last_row_every_country = gb.tail(1).sort_values("CountryCode")

y_3weeks = df["prediction_cat_14"]
y_5weeks = df["prediction_cat_28"]
y_cur = df["prediction_cat"]
df.drop(columns=["CountryCode", "prediction_cat_14", "prediction_cat_28"], inplace=True)


def load_data():
    X = df
    for (columnName, columnData) in X.iteritems():
        X[columnName] = X[columnName].fillna(-1)
    # separate X and y - X is the features and y is the result
    X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, y_3weeks, test_size=0.2)
    X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X, y_5weeks, test_size=0.2)
    all_data_sets = [[X_train_3, X_test_3, y_train_3, y_test_3],
                     [X_train_5, X_test_5, y_train_5, y_test_5]]
    return all_data_sets


def train_model():
    all_data_sets = load_data()
    clf = RandomForestClassifier(n_estimators=100)
    list_of_models = [clf]
    fitted_models = []
    for model_index in range(len(list_of_models)):
        for data_set in all_data_sets:
            model = list_of_models[model_index]
            model.fit(data_set[0], data_set[2])
            plot_confusion_matrix(model, data_set[1], data_set[3],
                                  cmap=plt.cm.get_cmap("Blues"),
                                  normalize='true')
            plt.show()
            fitted_models.append(model)
    return fitted_models


def finalle(prediction):
    with open('Countries_dict.json') as file:
        dictionary = json.load(file)
    prediction['cur_prediction_h'] = prediction['cur_prediction']
    prediction['prediction_14_h'] = prediction['prediction_14']
    prediction['prediction_28_h'] = prediction['prediction_28']

    for country in prediction['CountryCode'].unique():
        try:
            prediction.loc[prediction['CountryCode']==country,
                      'cur_prediction_h']=dictionary[country][str(prediction.loc[prediction['CountryCode']==country,
                      'cur_prediction']/2)[0]]
            prediction.loc[prediction['CountryCode'] == country,
                          'prediction_14_h'] = dictionary[country][str(prediction.loc[prediction['CountryCode']==country,
                          'prediction_14']/2)[0]]
            prediction.loc[prediction['CountryCode'] == country,
                          'prediction_28_h'] = dictionary[country][str(prediction.loc[prediction['CountryCode']==country,
                          'prediction_28']/2)[0]]
        except:
            prediction.loc[prediction['CountryCode'] == country,
                           'cur_prediction_h'] = ""
            prediction.loc[prediction['CountryCode'] == country,
                           'prediction_14_h'] = ''
            prediction.loc[prediction['CountryCode'] == country,
                           'prediction_28_h'] = ''
    info = pd.read_csv('travel_info_updated.csv')
    prediction = prediction.merge(info,on='CountryCode',how = 'left')
    return prediction


def predict():
    # fitted_models = train_model()
    # model_14, model_28 = fitted_models[0], fitted_models[1]
    # countries_predictions = pd.DataFrame(last_row_every_country["CountryCode"])
    # final_data = last_row_every_country
    # final_data.drop(columns=["CountryCode", "prediction_cat_14","prediction_cat_28"], inplace=True)
    # prediction_14, prediction_28 = model_14.predict(final_data),\
    #                                model_28.predict(final_data)
    #
    # countries_predictions["prediction_14"] = prediction_14
    # countries_predictions["prediction_28"] = prediction_28
    # countries_predictions["cur_prediction"] = y_cur
    # countries_predictions.to_csv("countries_predictions.csv")
    countries_predictions = pd.read_csv("countries_predictions.csv")
    countries_predictions = finalle(countries_predictions)
    return countries_predictions


def merge_with_geographical():
    countries = predict()
    shapefile = 'ne_110m_admin_0_countries.shp'
    gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
    gdf.columns = ['country', 'country_code', 'geometry']
    merged = gdf.merge(countries, left_on='country_code',
                       right_on='CountryCode')

    merged_json = json.loads(merged.to_json())
    json_data = json.dumps(merged_json)
    return json_data


if __name__ == '__main__':
    merge_with_geographical()
