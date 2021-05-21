from bokeh.layouts import row
from bokeh.plotting import curdoc
from bokeh.models.widgets import RadioButtonGroup
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, HoverTool
from models import merge_with_geographical
from bokeh.palettes import brewer


json_data = merge_with_geographical()

# Input GeoJSON source that contains features for plotting.
geosource = GeoJSONDataSource(geojson=json_data)

# Define a sequential multi-hue color palette.
palette = brewer["Oranges"][8]

# Reverse color order so that dark blue is highest obesity.
palette = palette[::-1]

# Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
color_mapper = LinearColorMapper(palette=palette, low=0, high=3)

# Add hover tool
hover_cur = HoverTool()

hover_cur.tooltips = """
<strong>Country</strong>: @country <br>
<strong>Open For Vaccinated</strong>: @Vaccinated <br>
<strong>Accommondation</strong>: @Accommondation <br>
<strong>Attractions</strong>: @Attractions <br>
<strong>Facilities</strong>: @Facilities <br>
<strong>Food</strong>: @Food <br>
<strong>Events</strong>: @Events <br>
@cur_prediction_h
"""

hover_14 = HoverTool()
hover_14.tooltips = """
<strong>Country</strong>: @country <br>
@prediction_14_h
"""

hover_28 = HoverTool()
hover_28.tooltips = """
<strong>Country</strong>: @country <br>
@prediction_28_h
"""

# Create figure object for current prediction
p_cur = figure(title='Current Prediction', plot_height=550, plot_width=950,
               toolbar_location=None, tools=[hover_cur], name='cur')
p_cur.xgrid.grid_line_color = None
p_cur.ygrid.grid_line_color = None
p_cur.axis.visible = False
p_cur.background_fill_color = "#bbe4f0"
# Add patch renderer to figure.
p_cur.patches('xs', 'ys', source=geosource,
              fill_color={'field': 'cur_prediction',
                          'transform': color_mapper},
              line_color='white', line_width=0.35, fill_alpha=1)

# Create figure object for 14 days prediction
p_14 = figure(title='Prediction for 14 days', plot_height=550, plot_width=950,
              toolbar_location=None, tools=[hover_14], name='pred_14')
p_14.xgrid.grid_line_color = None
p_14.ygrid.grid_line_color = None
p_14.axis.visible = False
p_14.background_fill_color = "#bbe4f0"
# Add patch renderer to figure.
p_14.patches('xs', 'ys', source=geosource,
             fill_color={'field': 'prediction_14', 'transform': color_mapper},
             line_color='white', line_width=0.35, fill_alpha=1)

# Create figure object for 28 days prediction
p_28 = figure(title='Prediction for 28 days' ,plot_height=550, plot_width=950,
              toolbar_location=None, tools=[hover_28], name='pred_28')
p_28.xgrid.grid_line_color = None
p_28.ygrid.grid_line_color = None
p_28.axis.visible = False
p_28.background_fill_color = "#bbe4f0"
# Add patch renderer to figure.
p_28.patches('xs', 'ys', source=geosource,
             fill_color={'field': 'prediction_28', 'transform': color_mapper},
             line_color='white', line_width=0.35, fill_alpha=1)
p_14.visible = False
p_28.visible = False


def update(attr, old, new):
    if old == 0 or not old:
        p_cur.visible = False
        if new == 1:
            p_14.visible = True
        else:
            p_28.visible = True
    elif old == 1:
        p_14.visible = False
        if new == 0:
            p_cur.visible = True
        else:
            p_28.visible = True
    else:
        p_28.visible = False
        if new == 0:
            p_cur.visible = True
        else:
            p_14.visible = True


LABELS = ["Current", "14-Days", "28-Days"]

radio_button_group = RadioButtonGroup(labels=LABELS, active=0)
radio_button_group.on_change('active', update)

curdoc().add_root(row(radio_button_group, p_cur, p_14, p_28 , name="main"))
