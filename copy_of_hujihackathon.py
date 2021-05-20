# -*- coding: utf-8 -*-
"""Copy of HujiHackathon.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lwYDQgrcum4rX_dy0qQgsx7fqsbBqapP
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
!pip install geopandas

import geopandas as gpd
shapefile = 'ne_110m_admin_0_countries.shp'

gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
gdf.columns = ['country', 'country_code', 'geometry']
gdf.head()





import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# load data
DATA_PATH = "data_for_learning.csv"
COLS_TO_REMOVE = ["CountryCode", "prediction_cat", "prediction_cat_14", "prediction_cat_28"]
df = pd.read_csv('data_for_learning.csv')
y_3weeks = df["prediction_cat_14"]
y_5weeks = df["prediction_cat_28"]
y_cur = df["prediction_cat"]


for (columnName, columnData) in df.iteritems():
  df[columnName] = df[columnName].fillna(-1)

print(df.isnull().sum().sum())

X = df.drop(columns=COLS_TO_REMOVE)

# separate X and y - X is the features and y is the result
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, y_3weeks, test_size=0.2)
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X, y_5weeks, test_size=0.2)
all_data_sets = [[X_train_3, X_test_3, y_train_3, y_test_3],
                 [X_train_5, X_test_5, y_train_5, y_test_5]]


# # cnn model
# cnn_model = tf.keras.models.Sequential()
# cnn_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# cnn_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# cnn_model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))
# # TODO: maybe change the loss to MSE
# cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# random forest model
clf = RandomForestClassifier(n_estimators=100)


"""
    TESTING
"""
list_of_models = [clf]
fitted_models = []
for model_index in range(len(list_of_models)):
    for data_set in all_data_sets:
        model = list_of_models[model_index]
        # if model_index == 0:
        #     continue
        #     # model.fit(x=data_set[0], y=data_set[2], batch_size=60,
        #     #           verbose=1, epochs=200, validation_split=0.25)
            
        #     # predicted_y = model.predict(data_set[1])
        #     # # tf.math.confusion_matrix(data_set[3], predicted_y)
        #     # plot_confusion_matrix(model, data_set[1], data_set[3],
        #     #                       cmap=plt.cm.get_cmap("Blues"),
        #     #                       normalize=True)
      #     # plt.show()
    # else:
        print(data_set[0].shape)
        model.fit(data_set[0], data_set[2])
        predicted_y = model.predict(data_set[1])
        # plot_confusion_matrix(model, data_set[1], data_set[3],
                              # cmap=plt.cm.get_cmap("Blues"), normalize='true')
        # plt.show()
        fitted_models.append(model)

### predict the last day of every country ####
import pandas as pd

COLS_TO_REMOVE = ["CountryCode", "prediction_cat", "prediction_cat_14", "prediction_cat_28"]

gb = df.groupby("CountryCode")
last_row_every_country = gb.tail(1).sort_values("CountryCode")
country_codes = pd.DataFrame(last_row_every_country["CountryCode"])
final_data = last_row_every_country.drop(columns=COLS_TO_REMOVE, axis=1)
model_14, model_28 = fitted_models[0], fitted_models[1]
prediction_14, prediction_28 = model_14.predict(final_data), model_28.predict(final_data)
country_codes["prediction_14"] = prediction_14
country_codes["prediction_28"] = prediction_28
country_codes["cur_prediction"] = y_cur

merged = gdf.merge(country_codes, left_on='country_code', right_on='CountryCode')
import json
merged_json = json.loads(merged.to_json())
json_data = json.dumps(merged_json)
from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, HoverTool
from bokeh.palettes import brewer



#Input GeoJSON source that contains features for plotting.
geosource = GeoJSONDataSource(geojson = json_data)

#Define a sequential multi-hue color palette.
palette = ("#990000", "#FFCC33", "#f5ee64", "#66FF99")

#Reverse color order so that dark blue is highest obesity.
palette = palette[::-1]

#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
color_mapper = LinearColorMapper(palette = palette, low = 0, high = 3)

#Define custom tick labels for color bar.
tick_labels = {'0': '0%', '5': '5%', '10':'10%', '15':'15%', '20':'20%', '25':'25%', '30':'30%','35':'35%', '40': '>40%'}

#Add hover tool
hover = HoverTool(tooltips = [ ('Country/region','@country')])


#Create color bar. 
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)

#Create figure object for current prediction
p_cur = figure(title = 'Current Prediction', plot_height = 600 , plot_width = 950, toolbar_location = None, tools = [hover])
p_cur.xgrid.grid_line_color = None
p_cur.ygrid.grid_line_color = None
p_cur.axis.visible = False
#Add patch renderer to figure. 
p_cur.patches('xs','ys', source = geosource,fill_color = {'field' :'cur_prediction', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)


#Create figure object for 14 days prediction
p_14 = figure(title = 'Prediction for 14 days', plot_height = 600 , plot_width = 950, toolbar_location = None, tools = [hover])
p_14.xgrid.grid_line_color = None
p_14.ygrid.grid_line_color = None
p_14.axis.visible = False
#Add patch renderer to figure. 
p_14.patches('xs','ys', source = geosource,fill_color = {'field' :'prediction_14', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)


#Create figure object for 28 days prediction
p_28 = figure(title = 'Prediction for 28 days', plot_height = 600 , plot_width = 950, toolbar_location = None, tools = [hover])
p_28.xgrid.grid_line_color = None
p_28.ygrid.grid_line_color = None
p_28.axis.visible = False
#Add patch renderer to figure. 
p_28.patches('xs','ys', source = geosource,fill_color = {'field' :'prediction_28', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)




#Specify figure layout.
# p.add_layout(color_bar, 'below')

#Display figure inline in Jupyter Notebook.
# output_notebook()

#Display figure.
#show(p_cur)

from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.plotting import curdoc
from bokeh.models.widgets import Button, Toggle, CheckboxGroup
from bokeh.models import Column


def show_pcur():
  show(p_cur)

def show_p14():
  show(p_14)

def show_p28():
  show(p_28)

button1 = Button(label = "current prediction")
button1.on_click(show_pcur)

button2 = Button(label = "14 days prediction")
button2.on_click(show_p14)

button3 = Button(label="28 days prediction")
button3.on_click(show_p28)

a = curdoc()


output_notebook()
show(Column(button1,button2,button3))

!bokeh serve --show