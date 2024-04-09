# Get data of the selected event;
def get_event_data(event_id: int, hall_id: str, data_location: str, dataset_name:str):

  import numpy as np
  import pandas as pd

  # event_id = 4
  #data_location = ""

  if hall_id == 'A':
    column_names=["evtid", "x", "y", "z", "track_number"]
    df_0 = pd.read_csv(data_location + dataset_name, sep=" ")  

  if hall_id == 'D':
    column_names=["evtid", "track_number", "x", "y", "covxx", "covxy", "covyy", "z", "rotation_angle", "drift_time"]
    df_0 = pd.read_csv(data_location + dataset_name, sep=" ", header=None)  
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
def track_visualization(event_id: int, hall_id: str, data_location: str, dataset_name:str):
   
    import numpy as np
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    import plotly.graph_objects as go
    
    df_test_0, df_test_1 = get_event_data(event_id, hall_id, data_location, dataset_name)

    df_0 = df_test_0 # df for track-1 
    df_1 = df_test_1 # df for track-2 

    x_array_0 = df_0['x']
    y_array_0 = df_0['y']
    z_array_0 = df_0['z']

    x_array_1 = df_1['x']
    y_array_1 = df_1['y']
    z_array_1 = df_1['z']

    viz_coordinates_2D_0 = np.array([x_array_0, y_array_0]).T # coordinate points for track-1
    viz_coordinates_2D_1 = np.array([x_array_1, y_array_1]).T # coordinate points for track-2

    ## PLOTTING 
    trace_1 = go.Scatter(
        x=viz_coordinates_2D_0[:, 0],
        y=viz_coordinates_2D_0[:, 1],
        mode='lines+markers',  # 'lines' for lines only, 'lines+markers' for lines with markers
        line=dict(color='cyan', width=2),  # line color and width
        marker=dict(size=8, color='blue', symbol='circle'),  # marker properties
        name='Line Plot'  # legend label
    )

    trace_2 = go.Scatter(
        x=viz_coordinates_2D_1[:, 0],
        y=viz_coordinates_2D_1[:, 1],
        mode='lines+markers',  # 'lines' for lines only, 'lines+markers' for lines with markers
        line=dict(color='green', width=2),  # line color and width
        marker=dict(size=8, color='orange', symbol='circle'),  # marker properties
        name='Line Plot'  # legend label
    )

    fig = go.Figure(data=[
    #scatter_points_0,
    #scatter_points_1,
    trace_1,
    trace_2
    ])


    if hall_id == 'A':
        title={

            'text': f"GMn (2 Track Stack) data - Event #{event_id}",
            'y': 0.94,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
            'size': 20,
            }
            },

    if hall_id == 'D':
        title={

            'text': f"Simon Taylor Hall-D data - Event #{event_id}",
            'y': 0.94,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
            'size': 20,
            }
            },


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

    title={
        'text': f"Simon Taylor Hall-D data - Event #{event_id}",
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