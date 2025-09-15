import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from surprise.model_selection import train_test_split, cross_validate
from surprise import Dataset, Reader, accuracy
from surprise.prediction_algorithms import SVD, SVDpp, NMF
import random
from PIL import Image, ImageDraw
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# a function for plotting the histogram of total number of labels by date, hour or weekday 
# colored by a column_name which can be maybe user_id, user_email, user_name
def plot_user_label_graph(app, df, column_name):
    app.layout = html.Div([
    
        # dash core components radio items
        dcc.RadioItems(
                id='date-type',
                options=[
                    {'label': 'Date', 'value': '%D'},
                    {'label': 'Hour', 'value': '%H'},
                    {'label': 'Weekday', 'value': '%A'}
                ],
                labelStyle={'display': 'inline-block'},
                value='%D',
            ),
        
        dcc.Graph(id='user-label'),
    ])

    # app callback -> to control the output of our 'user-label' graph by date_type
    @app.callback(Output('user-label', 'figure'), [Input('date-type', 'value')])
    def update_user_label_graph(date_type_value):
        
        # getting date based on our desired date_type
        x = df['created_on'].apply(lambda x: x.strftime(date_type_value))
        
        fig = go.Figure(px.histogram(x=x, y=df['label'], color=df[column_name]))
        fig.update_traces(opacity=0.75)
        fig.update_layout(
            title_text='Total number of labels by Hour'.upper(),
            xaxis_title_text='HOUR', 
            yaxis_title_text='FREQUENCY', 
            bargap=0.2, 
            bargroupgap=0.1,
            title_x=0.5)
        
        return fig
# Running k-fold cross validation using SVD on our data
def return_rmse_of_svd_kfold(df, columns, cv=5):
    accuracy_models = dict()
    for db_name in tqdm(columns):
        reader = Reader(rating_scale=(1,5))
        db_name_df = df[['user_id', 'item_id', db_name]]
        data = Dataset.load_from_df(db_name_df, reader)
        algo = SVD(n_factors=5)
        output = cross_validate(algo, data, measures=['RMSE'], cv=cv, verbose=False)
        accuracy_models[db_name] = output['test_rmse']
    return accuracy_models

# Shuffling a proportion of a dataframe's column
def shuffle_portion(_df, percentage=50): 
    arr = _df[_df.columns[-1]].values
    shuf = np.random.choice(np.arange(arr.shape[0]),  
                            round(arr.shape[0]*percentage/100), 
                            replace=False) 
    arr[np.sort(shuf)] = arr[shuf] 
    _df['label'] = arr
    return _df

# Comparing SVD before/after shuffling the data
def run_svd(df_db_name, num_of_users, is_shuffle=False, user_id=1):
    output_df  = pd.DataFrame(np.zeros((num_of_users,3)), columns=['user_id', 'rmse', 'is_shuffled'])
    reader = Reader(rating_scale=(1,5))
    if is_shuffle:
        # Shuffling a specific user
        df_db_name[df_db_name['user_id'] == user_id] = \
            df_db_name[df_db_name['user_id'] == user_id].groupby("user_id", as_index=False).apply(shuffle_portion)

    for k_time in range(num_of_users):

        data = Dataset.load_from_df(df_db_name, reader)
        train, test = train_test_split(data, test_size=0.1, random_state=k_time)
        algo = SVD(n_factors=5, random_state=k_time)
        algo.fit(train)
        pred = algo.test(test)
        output_df.iloc[k_time] = [str(user_id), accuracy.rmse(pred, verbose=False), str(int(is_shuffle))]
    
    return output_df, df_db_name

# how much a random input can affect the system, using shuffling and svd
def show_svd_with_without_shuffling(df, nrows, ncols, num_of_users):
    
    fig, ax = plt.subplots(nrows, ncols, figsize=(18,12))
    user_index = 0
    for index_zero in range(nrows):
        for index_one in range(ncols):
            user = sorted(df['user_id'].unique())[user_index]
            rmse_with_shuffling, _ = run_svd(df.copy(), num_of_users, is_shuffle=True, user_id=user)
            rmse_without_shuffling, _ = run_svd(df.copy(), num_of_users, user_id=user)
            mine = pd.concat([rmse_with_shuffling, rmse_without_shuffling])
            sns.boxplot(x="user_id", y="rmse",
                        hue="is_shuffled", palette='Set2',
                        data=mine, ax=ax[index_zero, index_one], width=.6)
            user_index += 1

    plt.show()

# run_time distribution plot
def show_run_time_dist(df, nrows, ncols, thresh=30):
    fig, axes = plt.subplots(nrows,ncols, figsize=(16,14))
    user_index = 0
    for row in range(nrows):
        for col in range(ncols):
            user = sorted(df['user_id'].unique())[user_index]
            threshold_time = df[df['user_id'] == user]['insert_time'].sort_values().diff().dt.seconds
            ax = threshold_time[threshold_time < thresh].plot(kind='hist', ax=axes[row][col], title=user)
            ax.set_xlim([0,thresh])
            user_index += 1
    plt.show()

# normal vs comparative user/random user plot
def normal_vs_comparative_user_plot(df, df_compare, metric='trustworthy'):
    
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    user_dict = {}
    for user in df_compare['user_id'].unique():
        mydict = { 'tp': [], 'fp': [], 'range': []}
        for cmpr_index, cmpr_row in df_compare.iterrows():
            if user in df['user_id'].unique():
                specific_user = df[df['user_id'] == user]
                im1_value = (specific_user[specific_user['data_image_part'] == cmpr_row['im1_path']][metric]).values
                im2_value = (specific_user[specific_user['data_image_part'] == cmpr_row['im2_path']][metric]).values
                if  (im1_value >= im2_value) and (cmpr_row[metric]==1):
                    mydict['tp'].append(1)
                    mydict['fp'].append(0)
                    mydict['range'].append((im1_value[0] - im2_value[0]))
                elif (im1_value > im2_value) and (cmpr_row[metric]==2):
                    mydict['fp'].append(1)
                    mydict['tp'].append(0)
                    mydict['range'].append((im1_value[0] - im2_value[0]))
                elif (im1_value <= im2_value) and (cmpr_row[metric]==2):
                    mydict['tp'].append(1)
                    mydict['fp'].append(0)
                    mydict['range'].append((im1_value[0] - im2_value[0]))
                elif (im1_value < im2_value) and (cmpr_row[metric]==1):
                    mydict['fp'].append(1)
                    mydict['tp'].append(0)
                    mydict['range'].append((im1_value[0] - im2_value[0]))

        user_dict[user] = mydict
        
    return user_dict

# Returning t-sne image viewer
def return_tsne_image_viewer(tx, ty, embedding, images, df):
    
    width = 4000
    height = 3000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    for img, x, y in zip(images, tx, ty):
        tile = Image.open(img)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    return full_image
