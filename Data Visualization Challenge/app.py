import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
import openai


# Sample data for demonstration (replace with your own DataFrame)
df = pd.read_csv('data/cleaned_data.csv')

# Top 5 unique values for 'Genus', 'Species', 'Common Name'
top_5_genus = df['Genus'].value_counts().head(5).index.tolist()
top_5_species = df['Species'].value_counts().head(5).index.tolist()
top_5_common_name = df['Common Name'].value_counts().head(5).index.tolist()

# App layout
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H4(
        "Buffalization Data Visualization Challenge",
        style={
            'textAlign': 'center', 
            'fontSize': '32px', 
            'fontWeight': 'bold', 
            'color': '#ffffff', 
            'textShadow': '3px 3px 6px rgba(0, 0, 0, 0.8)',
            'marginBottom': '20px'
        }
    ),

    html.P("Select Categorical Feature:", style={
            'fontSize': '20px', 
            'fontWeight': 'bold', 
            'color': '#ffffff', 
            'textShadow': '3px 3px 6px rgba(0, 0, 0, 0.8)',
            'marginBottom': '20px'
        }),
    dcc.Dropdown(
        id='categorical-feature',
        options=[
            {'label': 'Genus', 'value': 'Genus'},
            {'label': 'Species', 'value': 'Species'},
            {'label': 'Common Name', 'value': 'Common Name'},
            {'label': 'Tree Type', 'value': 'Tree Type'}
        ],
        value='Genus',
        clearable=False,
        className="custom-dropdown"  # Add class for CSS
    ),

    html.P("Select Values for Categorical Feature:", style={
            'fontSize': '20px', 
            'fontWeight': 'bold', 
            'color': '#ffffff', 
            'textShadow': '3px 3px 6px rgba(0, 0, 0, 0.8)',
            'marginBottom': '20px'
        }),
    dcc.Dropdown(
        id='value-dropdown',
        value=top_5_genus[0],
        multi=True,
        clearable=False,
        className="custom-dropdown"
    ),
    html.P("Select the metric:", style={
            'fontSize': '20px', 
            'fontWeight': 'bold', 
            'color': '#ffffff', 
            'textShadow': '3px 3px 6px rgba(0, 0, 0, 0.8)',
            'marginBottom': '20px'
        }),
    dcc.Dropdown(
        id='metric-dropdown',
        options=[
            {'label': "Height", 'value': "Height"},
            {'label': 'Canopy Spread', 'value': 'Canopy Spread'}
        ],
        value="Height",
        clearable=False,
        className="custom-dropdown"  # Add class for CSS
    ),

    html.Div([
        html.Div([dcc.Graph(id="graph", className="custom-graph")], style={'flex': 1, 'padding': '20px'}),
        html.Div([dcc.Graph(id="pie-chart", className="custom-graph")], style={'flex': 1, 'padding': '20px'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}),

   html.Div([
    html.Div([
        dcc.Graph(id="scatter-plot", className="custom-graph", style={'width': '100%'})
    ], style={'flex': 1, 'padding': '20px'}),

    html.Div([
        dcc.Graph(id="scatter-plot-overall", className="custom-graph", style={'width': '100%'})
    ], style={'flex': 1, 'padding': '20px'}),
], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}),
    
html.Div([
    html.Div([
        dcc.Graph(id="geo-plot", className="custom-graph", style={'width': '100%'})
    ], style={'padding': '20px', 'flex': 1}),
    
    html.Div([
        dcc.Graph(id="geo-plot-overall", className="custom-graph", style={'width': '100%'})
    ], style={'padding': '20px', 'flex': 1}),
], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}),
html.Div([
    html.Div([
        dcc.Graph(id="scatter", className="custom-graph", style={'width': '100%'})
    ], style={'flex': 1, 'padding': '20px'}),

    html.Div([
        dcc.Graph(id="scatter-overall", className="custom-graph", style={'width': '100%'})
    ], style={'flex': 1, 'padding': '20px'}),
], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}),


])  # Set dark background




# Function to read the API key from the file
def read_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()


# Set up the OpenAI API
openai.api_key = read_api_key('data/apikey.txt')


def generate_insights(df, categorical_feature, selected_values):
    # Prepare the data for analysis by the LLM
    subset_df = df[df[categorical_feature].isin(selected_values)]
    feature_summary = subset_df.describe(include='all').to_string()

    # Create a prompt for the LLM
    prompt = f"Analyze the following data and provide insights in an engaging and visually appealing way:\n\n{feature_summary}\n\n"
    # List available models
    models = openai.Model.list()
    # Query the LLM to generate insights
    response = openai.Completion.create(
        engine= 'gpt-3.5-turbo',  # You can use other models such as GPT-3 or GPT-4
        prompt=prompt,
        max_tokens=300,
        temperature=0.7
    )
    insights = response.choices[0].text.strip()
    return insights
@app.callback(
    [Output('value-dropdown', 'options'),
     Output('value-dropdown', 'value')],
    [Input('categorical-feature', 'value')]
)
def update_value_dropdown(categorical_feature):
    if categorical_feature == 'Genus':
        options = [{'label': i, 'value': i} for i in top_5_genus]
        value = [top_5_genus[0], top_5_genus[1]]  # Default values
    elif categorical_feature == 'Species':
        options = [{'label': i, 'value': i} for i in top_5_species]
        value = [top_5_species[0], top_5_species[1]]  # Default values
    elif categorical_feature == 'Common Name':
        options = [{'label': i, 'value': i} for i in top_5_common_name]
        value = [top_5_common_name[0], top_5_common_name[1]]  # Default values
    else:  # Tree Type
        options = [{'label': i, 'value': i} for i in df['Tree Type'].unique()]
        value = [df['Tree Type'].unique()[0], df['Tree Type'].unique()[1]]  # Default values

    return options, value

@app.callback(
    [Output('metric-dropdown', 'options'),
     Output('metric-dropdown', 'value')],
    [Input('categorical-feature', 'value')]
)
def update_metric_dropdown(categorical_feature):
    # Default options for the metric dropdown
    options = [
        {'label': 'Height', 'value': 'Height'},
        {'label': 'Canopy Spread', 'value': 'Canopy Spread'}
    ]
    
    # You can modify the options based on the selected categorical feature, if needed
    # For now, returning default values as it is always 'Height' and 'Canopy Spread'
    value = 'Height'  # Default value

    return options, value

@app.callback(
    [Output('graph', 'figure'),
     Output('pie-chart', 'figure')],
    Input('categorical-feature', 'value'),
    Input('value-dropdown', 'value'),
    Input('metric-dropdown', 'value')

)
def update_graph(categorical_feature, selected_values,metric):
    print(f"Selected values: {selected_values}")
    print(f"metric: {metric}")
    
    # Filter the dataframe
    filtered_df = df[df[categorical_feature].isin(selected_values)]

    # Check for valid data
    if filtered_df.empty or filtered_df[metric].isnull().all():
        return go.Figure(), go.Figure()  # Return empty figures if no valid data

    # Nature-inspired color palette
    nature_colors = ['#2E7D32', '#66BB6A', '#A1887F', '#1E88E5', '#8D6E63']  # Greens, browns, and blues

    # Histogram with updated colors
    fig = px.histogram(
        filtered_df, x=metric, color=categorical_feature,
        title=f"🌿 {metric} Distribution by {categorical_feature}",
        labels={metric: "metric ", categorical_feature: f"{categorical_feature} Category"},
        marginal="box", nbins=20,
        color_discrete_sequence=nature_colors
    )

    # KDE using scipy.stats.gaussian_kde
    metric_data = filtered_df[metric].dropna()

    kde = gaussian_kde(metric_data)
    x_vals = np.linspace(metric_data.min(), metric_data.max(), 1000)
    y_vals = kde(x_vals)

    # Smooth KDE line in dark green
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', 
                             name="KDE", line=dict(color='#145A32', width=3, dash="solid")))

    # Apply updated nature-styled layout
    fig.update_layout(
        xaxis_title="metric ",
        yaxis_title="Count",
        title_font=dict(size=20, color="#1B5E20"),
        bargap=0.15,
        plot_bgcolor='rgba(255,255,255,0)',  # Transparent background
        paper_bgcolor='#E8F5E9',  # Soft pastel green background
        font=dict(family="Arial", size=14, color="#1B5E20"),
        xaxis=dict(gridcolor='rgba(50,50,50,0.2)', zerolinecolor='rgba(50,50,50,0.2)'),
        yaxis=dict(gridcolor='rgba(50,50,50,0.2)', zerolinecolor='rgba(50,50,50,0.2)'),
        showlegend=True,
    )

    # Pie chart using nature colors
    size = filtered_df[categorical_feature].value_counts()
    labels = filtered_df[categorical_feature].value_counts().index
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=size, hole=0.3)])

    fig_pie.update_traces(
        marker=dict(colors=nature_colors, line=dict(color='#1B5E20', width=1)),
        textinfo='label+percent'
    )

    fig_pie.update_layout(
        title=f"🌍 {categorical_feature} Distribution",
        title_font=dict(size=20, color="#1B5E20"),
        paper_bgcolor='#E8F5E9',  # Light green background
        font=dict(family="Arial", size=14, color="#1B5E20"),
    )
    
    return fig, fig_pie



# Callback for scatter plot update (with new 'metric-dropdown')
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('categorical-feature', 'value'),
    Input('value-dropdown', 'value'),
    Input('metric-dropdown', 'value')  # New input for metric
)
def update_scatter_plot(categorical_feature, selected_values, metric):
    print(f"Selected values: {selected_values}, Metric: {metric}")
    
    # Filter the dataframe based on selected categorical feature and values
    filtered_df = df[df[categorical_feature].isin(selected_values)]
    
    # Check if the filtered dataframe has valid data
    if filtered_df.empty:
        return go.Figure()  # Return an empty figure if no valid data
    
    # Custom colormap based on metric
    metric_colormap = [
        [0.0, '#A7D129'],  # Light Green (Small Trees)
        [0.3, '#4CAF50'],  # Natural Green (Medium Trees)
        [0.6, '#1B5E20'],  # Deep Forest Green (Taller Trees)
        [0.9, '#8D6E63'],  # Brownish for Very Tall Trees
        [1.0, '#4E342E']   # Dark Brown (Oldest Trees)
    ]

    # Create scatter plot using Plotly Express
    fig = px.scatter(
        filtered_df,
        x='Longitude', 
        y='Latitude', 
        color=metric,
        size=metric,
        hover_data=['Genus', 'Species', 'Common Name'],
        color_continuous_scale=metric_colormap,  # Apply custom colormap
        animation_frame=metric,  # Adds animation based on the metric
        title=f"🌳 Tree Distribution by Location ({categorical_feature}) - {metric}",
        labels={"Longitude": "🧭 Longitude (°)", "Latitude": "🌍 Latitude (°)", metric: f"Tree {metric}"},
    )

    # Update layout with nature aesthetics
    fig.update_layout(
        title=f"🌍 Tree {metric} Distribution by Location ({categorical_feature})",
        title_font=dict(size=20, color="#1B5E20"),
        xaxis_title="🧭 Longitude (°)",
        yaxis_title="🌿 Latitude",
        paper_bgcolor='#E8F5E9',  # Light pastel green background
        plot_bgcolor='rgba(255,255,255,0)',  # Transparent plot area
        font=dict(family="Arial", size=14, color="#1B5E20"),
        xaxis=dict(gridcolor='rgba(50,50,50,0.2)', zerolinecolor='rgba(50,50,50,0.2)'),
        yaxis=dict(gridcolor='rgba(50,50,50,0.2)', zerolinecolor='rgba(50,50,50,0.2)'),
        showlegend=True
    )

    return fig


# Callback for scatter plot update (plotting for entire df with new 'metric-dropdown')
@app.callback(
    Output('scatter-plot-overall', 'figure'),
    Input('categorical-feature', 'value'),
    Input('value-dropdown', 'value'),
    Input('metric-dropdown', 'value')  # New input for metric
)
def update_scatter_plot_overall(categorical_feature, selected_values, metric):
    # No need for filtering, just use the entire df
    filtered_df = df
    
    # Check if filtered dataframe has valid data
    if filtered_df.empty:
        return go.Figure()  # Return empty figure if no valid data
    
    # Custom colormap based on metric
    metric_colormap = [
        [0.0, '#A7D129'],  # Light Green (Small Trees)
        [0.3, '#4CAF50'],  # Natural Green (Medium Trees)
        [0.6, '#1B5E20'],  # Deep Forest Green (Taller Trees)
        [0.9, '#8D6E63'],  # Brownish for Very Tall Trees
        [1.0, '#4E342E']   # Dark Brown (Oldest Trees)
    ]

    # Create scatter plot using Plotly Express
    fig = px.scatter(
        filtered_df,
        x='Longitude', 
        y='Latitude', 
        color=metric,
        size=metric,
        hover_data=['Genus', 'Species', 'Common Name', 'Tree Type'],
        color_continuous_scale=metric_colormap,  # Apply custom colormap
        animation_frame=metric,  # Add animation based on metric or other columns
        title=f"🌳 Tree Distribution by Location (Entire Dataset) - {metric}",
        labels={"Longitude": "🧭 Longitude (°)", "Latitude": "🌍 Latitude (°)", metric: f"Tree {metric}"},
    )

    # Update layout with nature aesthetics
    fig.update_layout(
        title=f"🌳 Tree {metric} Distribution by Location (Entire Dataset)",
        title_font=dict(size=20, color="#2E7D32"),  # Forest Green color
        xaxis_title="🧭 Longitude (°)",
        yaxis_title="🌍 Latitude (°)",
        paper_bgcolor='#E8F5E9',  # Softer pastel green background (lighter shade)
        plot_bgcolor='rgba(255,255,255,0)',  # Transparent plot area
        font=dict(family="Arial", size=14, color="#2E7D32"),
        xaxis=dict(gridcolor='rgba(50,50,50,0.2)', zerolinecolor='rgba(50,50,50,0.2)'),
        yaxis=dict(gridcolor='rgba(50,50,50,0.2)', zerolinecolor='rgba(50,50,50,0.2)'),
        showlegend=True
    )
    
    return fig




# Callback for geo plot update with animation
# Callback for geo plot update with animation
# Callback for geo plot update with animation and pause functionality
# Callback for geo plot update with dynamic zoom based on latitude and longitude data points
@app.callback(
    Output('geo-plot', 'figure'),
    Input('categorical-feature', 'value'),
    Input('value-dropdown', 'value'),
    Input('metric-dropdown', 'value')  # New metric input
)
def update_geo_plot(categorical_feature, selected_values, metric):
    print(f"Selected values: {selected_values}, Metric: {metric}")
    
    # Filter the dataframe to include only rows with selected categorical feature values
    filtered_df = df[df[categorical_feature].isin(selected_values)]
    
    # Check if filtered dataframe has valid data
    if filtered_df.empty or filtered_df['Longitude'].isnull().any() or filtered_df['Latitude'].isnull().any():
        return go.Figure()  # Return empty figure if no valid data
    
    # Calculate the bounding box based on the data points
    min_lat = filtered_df['Latitude'].min()
    max_lat = filtered_df['Latitude'].max()
    min_lon = filtered_df['Longitude'].min()
    max_lon = filtered_df['Longitude'].max()
    
    # Compute the center of the data points
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # Calculate the zoom level based on the range of the latitude and longitude
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    max_range = max(lat_range, lon_range)  # Use the larger range to determine zoom
    
    # Set zoom level; adjust for better fitting
    if max_range < 0.1:
        zoom_level = 12  # Highly zoomed in
    elif max_range < 1:
        zoom_level = 10  # Moderate zoom
    elif max_range < 5:
        zoom_level = 8  # Wider zoom
    else:
        zoom_level = 6  # Very wide zoom
    
    # Create frames for animation (using the selected metric)
    frames = []
    for metric_value in sorted(filtered_df[metric].unique()):
        frame_data = filtered_df[filtered_df[metric] == metric_value]
        frames.append(go.Frame(
            data=[go.Scattermapbox(
                lat=frame_data['Latitude'],
                lon=frame_data['Longitude'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=frame_data[metric] / 2,  # Marker size based on selected metric
                    color=frame_data[metric],  # Color based on selected metric
                    colorscale='Viridis',
                    colorbar={'title': f'{metric}'},
                    opacity=0.8  # Set the opacity for the markers
                ),
                text=frame_data['Common Name'],  # Display the Common Name on hover
                hoverinfo='text+lat+lon',  # Show latitude, longitude, and Common Name on hover
            )],
            name=str(metric_value)  # Set frame name based on selected metric value
        ))

    # Create the initial plot
    fig = go.Figure(
        data=[go.Scattermapbox(
            lat=filtered_df['Latitude'],
            lon=filtered_df['Longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=filtered_df[metric] / 2,  # Marker size based on selected metric
                color=filtered_df[metric],  # Color based on selected metric
                colorscale='Viridis',
                colorbar={'title': f'{metric}'},  # Add metric name to colorbar
                opacity=0.8  # Set the opacity for the markers
            ),
            text=filtered_df['Common Name'],  # Display the Common Name on hover
            hoverinfo='text+lat+lon',  # Show latitude, longitude, and Common Name on hover
        )],
        layout=go.Layout(
            mapbox=dict(
                style="carto-positron",  # You can use different map styles like 'open-street-map', 'carto-positron', etc.
                center=dict(lat=center_lat, lon=center_lon),  # Set the map center dynamically based on data
                zoom=15,  # Adjust zoom level
                bearing=20,  # Add some rotation for cool animation effect
                pitch=45  # Add tilt for 3D effect
            ),
            title=f"🌳 Tree {metric} Distribution on Map for {categorical_feature} data ({metric})",
            title_font=dict(size=20, color="#2E7D32"),  # Forest Green color for title
            paper_bgcolor='#E8F5E9',  # Softer pastel green background (lighter shade)
            plot_bgcolor='rgba(255,255,255,0)',  # Transparent plot area
            font=dict(family="Arial", size=14, color="#2E7D32"),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=500, redraw=True),
                            fromcurrent=True,
                            mode='immediate',
                            transition=dict(duration=300)
                        )]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=True),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ]
            )],
            transition_duration=500,  # Smooth transition between animation frames
        ),
        frames=frames  # Add frames directly to the figure
    )
    
    return fig

@app.callback(
    Output('geo-plot-overall', 'figure'),
    Input('categorical-feature', 'value'),
    Input('value-dropdown', 'value'),
    Input('metric-dropdown', 'value')  # New metric input
)
def update_geo_plot_overall(categorical_feature, selected_values, metric):
    print(f"Selected values: {selected_values}, Metric: {metric}")
    
    # Filter the dataframe to include only rows with selected categorical feature values
    filtered_df = df
    
    # Check if filtered dataframe has valid data
    if filtered_df.empty or filtered_df['Longitude'].isnull().any() or filtered_df['Latitude'].isnull().any():
        return go.Figure()  # Return empty figure if no valid data
    
    # Calculate the bounding box based on the data points
    min_lat = filtered_df['Latitude'].min()
    max_lat = filtered_df['Latitude'].max()
    min_lon = filtered_df['Longitude'].min()
    max_lon = filtered_df['Longitude'].max()
    
    # Compute the center of the data points
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # Calculate the zoom level based on the range of the latitude and longitude
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    max_range = max(lat_range, lon_range)  # Use the larger range to determine zoom
    
    # Set zoom level; adjust for better fitting
    if max_range < 0.1:
        zoom_level = 12  # Highly zoomed in
    elif max_range < 1:
        zoom_level = 10  # Moderate zoom
    elif max_range < 5:
        zoom_level = 8  # Wider zoom
    else:
        zoom_level = 6  # Very wide zoom
    
    # Create frames for animation (using the selected metric)
    frames = []
    for metric_value in sorted(filtered_df[metric].unique()):
        frame_data = filtered_df[filtered_df[metric] == metric_value]
        frames.append(go.Frame(
            data=[go.Scattermapbox(
                lat=frame_data['Latitude'],
                lon=frame_data['Longitude'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=frame_data[metric] / 2,  # Marker size based on selected metric
                    color=frame_data[metric],  # Color based on selected metric
                    colorscale='Viridis',
                    colorbar={'title': f'{metric}'},  # Add metric name to colorbar
                    opacity=0.8  # Set the opacity for the markers
                ),
                text=frame_data['Common Name'],  # Display the Common Name on hover
                hoverinfo='text+lat+lon',  # Show latitude, longitude, and Common Name on hover
            )],
            name=str(metric_value)  # Set frame name based on selected metric value
        ))

    # Create the initial plot
    fig = go.Figure(
        data=[go.Scattermapbox(
            lat=filtered_df['Latitude'],
            lon=filtered_df['Longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=filtered_df[metric] / 2,  # Marker size based on selected metric
                color=filtered_df[metric],  # Color based on selected metric
                colorscale='Viridis',
                colorbar={'title': f'{metric}'},  # Add metric name to colorbar
                opacity=0.8  # Set the opacity for the markers
            ),
            text=filtered_df['Common Name'],  # Display the Common Name on hover
            hoverinfo='text+lat+lon',  # Show latitude, longitude, and Common Name on hover
        )],
        layout=go.Layout(
            mapbox=dict(
                style="carto-positron",  # You can use different map styles like 'open-street-map', 'carto-positron', etc.
                center=dict(lat=center_lat, lon=center_lon),  # Set the map center dynamically based on data
                zoom=15,  # Adjust zoom level
                bearing=20,  # Add some rotation for cool animation effect
                pitch=45  # Add tilt for 3D effect
            ),
            title=f"🌳 Tree {metric} Distribution on Map (Entire Dataset) ({metric})",
            title_font=dict(size=20, color="#2E7D32"),  # Forest Green color for title
            paper_bgcolor='#E8F5E9',  # Softer pastel green background (lighter shade)
            plot_bgcolor='rgba(255,255,255,0)',  # Transparent plot area
            font=dict(family="Arial", size=14, color="#2E7D32"),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=500, redraw=True),
                            fromcurrent=True,
                            mode='immediate',
                            transition=dict(duration=300)
                        )]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=True),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ]
            )],
            transition_duration=500,  # Smooth transition between animation frames
        ),
        frames=frames  # Add frames directly to the figure
    )
    
    return fig

@app.callback(
    Output('scatter', 'figure'),
    Input('categorical-feature', 'value'),
    Input('value-dropdown', 'value')
)
def update_tree_metric_vs_canopy(categorical_feature, selected_values):
    # No need for filtering, just use the entire df
    filtered_df = df[df[categorical_feature].isin(selected_values)]
    
    # Check if filtered dataframe has valid data
    if filtered_df.empty or filtered_df["Height"].isnull().any() or filtered_df['Canopy Spread'].isnull().any():
        return go.Figure()  # Return empty figure if no valid data
    
    # Create scatter plot using Plotly Express with interactive hover features
    fig = px.scatter(
        filtered_df,
        x="Height", 
        y="Canopy Spread", 
        title="🌳 Height vs. Canopy Spread",
        labels={"Height": "Height ", "Canopy Spread": "Canopy Spread "},
    )

    # Apply same color scheme as the previous plot, like green and brown shades
    metric_colormap = [
        [0.0, '#A7D129'],  # Light Green (Small Trees)
        [0.3, '#4CAF50'],  # Natural Green (Medium Trees)
        [0.6, '#1B5E20'],  # Deep Forest Green (Taller Trees)
        [0.9, '#8D6E63'],  # Brownish for Very Tall Trees
        [1.0, '#4E342E']   # Dark Brown (Oldest Trees)
    ]
    
    fig.update_traces(marker=dict(color=filtered_df["Height"], colorscale=metric_colormap))
    
    # Update layout with the same color scheme and style
    fig.update_layout(
        title=f"🌳 Height vs. Canopy Spread for {categorical_feature}",
        title_font=dict(size=20, color="#2E7D32"),  # Forest Green color
        xaxis_title="Tree metric",
        yaxis_title="Canopy Spread",
        paper_bgcolor='#E8F5E9',  # Softer pastel green background (lighter shade)
        plot_bgcolor='rgba(255,255,255,0)',  # Transparent plot area
        font=dict(family="Arial", size=14, color="#2E7D32"),
        xaxis=dict(gridcolor='rgba(50,50,50,0.2)', zerolinecolor='rgba(50,50,50,0.2)'),
        yaxis=dict(gridcolor='rgba(50,50,50,0.2)', zerolinecolor='rgba(50,50,50,0.2)'),
        showlegend=False
    )
    
    return fig
@app.callback(
    Output('scatter-overall', 'figure'),
    Input('categorical-feature', 'value'),
    Input('value-dropdown', 'value')
)
def update_tree_metric_vs_canopy(categorical_feature, selected_values):
    # No need for filtering, just use the entire df
    filtered_df = df
    
    # Check if filtered dataframe has valid data
    if filtered_df.empty or filtered_df["Height"].isnull().any() or filtered_df['Canopy Spread'].isnull().any():
        return go.Figure()  # Return empty figure if no valid data
    
    # Create scatter plot using Plotly Express with interactive hover features
    fig = px.scatter(
        filtered_df,
        x="Height", 
        y="Canopy Spread", 
        title="🌳 Height vs. Canopy Spread {Entire Dataset}",
        labels={"Height": "Height", "Canopy Spread": "Canopy Spread"},
    )

    # Apply same color scheme as the previous plot, like green and brown shades
    metric_colormap = [
        [0.0, '#A7D129'],  # Light Green (Small Trees)
        [0.3, '#4CAF50'],  # Natural Green (Medium Trees)
        [0.6, '#1B5E20'],  # Deep Forest Green (Taller Trees)
        [0.9, '#8D6E63'],  # Brownish for Very Tall Trees
        [1.0, '#4E342E']   # Dark Brown (Oldest Trees)
    ]
    
    fig.update_traces(marker=dict(color=filtered_df["Height"], colorscale=metric_colormap))
    
    # Update layout with the same color scheme and style
    fig.update_layout(
        title="🌳 Height vs. Canopy Spread (Entire Dataset)",
        title_font=dict(size=20, color="#2E7D32"),  # Forest Green color
        xaxis_title="Tree metric",
        yaxis_title="Canopy Spread",
        paper_bgcolor='#E8F5E9',  # Softer pastel green background (lighter shade)
        plot_bgcolor='rgba(255,255,255,0)',  # Transparent plot area
        font=dict(family="Arial", size=14, color="#2E7D32"),
        xaxis=dict(gridcolor='rgba(50,50,50,0.2)', zerolinecolor='rgba(50,50,50,0.2)'),
        yaxis=dict(gridcolor='rgba(50,50,50,0.2)', zerolinecolor='rgba(50,50,50,0.2)'),
        showlegend=False
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
