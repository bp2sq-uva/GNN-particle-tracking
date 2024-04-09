# Get data of the selected event
def get_event_data(event_id):

  import numpy as np
  import pandas as pd

  # event_id = 4
  data_location = ""
  
  column_names=["evtid", "track_number", "x", "y", "covxx", "covxy", "covyy", "z", "rotation_angle", "drift_time"]
  df_0 = pd.read_csv(data_location+"rho_ml_training.dat", sep=" ", header=None)
  df_0.columns = column_names

  try:
    if event_id in df_0['evtid']:

      df_event = df_0[df_0['evtid']==event_id] # df section with the selected event id
      track_list = df_event['track_number'].unique() # ids of the tracks in the selected event

      df_track_0 = df_0[df_0['evtid']==event_id][df_0[df_0['evtid']==event_id]['track_number']==track_list[0]] # track-1
      df_track_1 = df_0[df_0['evtid']==event_id][df_0[df_0['evtid']==event_id]['track_number']==track_list[1]] # track-2
      
      return df_track_0, df_track_1

  except TypeError as e:
    print('no such event')


# Plot 2 tracks in the selected event
def track_visualization(event_id):
    import numpy as np
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    import plotly.graph_objects as go

    df_test_0, df_test_1 = get_event_data(event_id)
    df_0 = df_test_0 # df for track-1 
    df_1 = df_test_1 # df for track-2 

    x_array_0 = df_0['x']
    y_array_0 = df_0['y']
    z_array_0 = df_0['z']

    x_array_1 = df_1['x']
    y_array_1 = df_1['y']
    z_array_1 = df_1['z']

    viz_coordinates_2D_0 = np.array([x_array_0, y_array_0, z_array_0]).T # coordinate points for track-1
    viz_coordinates_2D_1 = np.array([x_array_1, y_array_1, z_array_1]).T # coordinate points for track-2


    ## PLOTTING 

    # Create a scatter plot for the 2D coordinates of track-1
    scatter_points_0 = go.Scatter3d(
        x=viz_coordinates_2D_0[:, 0],
        y=viz_coordinates_2D_0[:, 1],
        z=viz_coordinates_2D_0[:, 2],
        mode='markers',
        marker=dict(size=3, color='red'),
        name='track 1'
    )

    # Create a scatter plot for the 2D coordinates of track-2
    scatter_points_1 = go.Scatter3d(
        x=viz_coordinates_2D_1[:, 0],
        y=viz_coordinates_2D_1[:, 1],
        z=viz_coordinates_2D_1[:, 2],
        mode='markers',
        marker=dict(size=3, color='green'),
        name='track 2'
    )

    # Create a scatter plot for the 2D coordinates of the origin at (0,0,0)
    scatter_points_origin = go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(size=10, color='black'),
        name='origin at (0,0,0)'
    )


    links_data_0 = [{'start_coords': viz_coordinates_2D_0[i], 
                'end_coords': viz_coordinates_2D_0[i+1],
                'color': 'cyan'} for i in range(0, len(viz_coordinates_2D_0)-1)]

    links_data_1 = [{'start_coords': viz_coordinates_2D_1[i], 
                'end_coords': viz_coordinates_2D_1[i+1],
                'color': 'orange'} for i in range(0, len(viz_coordinates_2D_1)-1)]


    # Define the links between specific points
    def create_link_trace(start_coords, end_coords, color):
        return go.Scatter3d(
            x=[start_coords[0], end_coords[0]],
            y=[start_coords[1], end_coords[1]],
            z=[start_coords[2], end_coords[2]],
            mode='lines',
            line=dict(color=color, width=2),
            name='',
            showlegend=False
        )


    link_traces_0 = [create_link_trace(link['start_coords'], link['end_coords'], link['color']) for link in links_data_0]
    link_traces_1 = [create_link_trace(link['start_coords'], link['end_coords'], link['color']) for link in links_data_1]


    fig = go.Figure(data=[
        scatter_points_0,
        scatter_points_1,
        scatter_points_origin,
        *link_traces_0,
        *link_traces_1
    ])
    
    # Link display buttons
    buttons = [
        {
            'method': 'update',
            'label': 'Show Links',
            'args': [{'visible': [True] * len(fig.data)}],
        },
        {
            'method': 'update',
            'label': 'Hide Links',
            'args': [{'visible': [True] * (len(fig.data) - len(link_traces_0) - len(link_traces_1)) + [False] * (len(link_traces_0) + len(link_traces_1))}],
        }
    ]

    fig.update_layout(

        scene=dict(
            xaxis=dict(
                showticklabels=False,
                linecolor='rgba(0,200,200,200)',  
                backgroundcolor='rgba(0,0,0,0)'  
            ),
            yaxis=dict(
                showticklabels=False,
                linecolor='rgba(0,200,200,200)',  
                backgroundcolor='rgba(0,0,0,0)'  
            ),
            zaxis=dict(
                showticklabels=False,
                linecolor='rgba(0,200,200,200)',  
                backgroundcolor='rgba(200,200,200,240)'  
            ),
        ),

        updatemenus=[
            {
                'buttons': buttons,
                'direction': 'down',
                'showactive': True,
                'x': 1.0,
                'xanchor': 'left',
                'y': 1.2,
                'yanchor': 'top',
            },
        ],

        title={
            'text': f"Simon Taylor (2 Track) data - Event #{event_id}",
            'y': 0.94,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 20,
            }
        },
        paper_bgcolor='white'
    )


    fig.show()