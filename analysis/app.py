# scp root@144.76.224.122:/home/jupyters/attractiveness/ml-cfd-analysis/notebooks/labels.pkl .

import dash
import json
from PIL import Image
import pandas as pd
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_classif as MI

from utils import query_labels, get_labels, is_running_local
import utils

# global vars are in capital letters
LABELS = get_labels(1000)
UINFO = 'email'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
main_vars = {
    'None': 'all',
    'Dataset': 'db_name',
    'User': UINFO,
    'Label': 'label'
}

date_time_dict = {
    '%D': 'Date',
    '%H': 'Hour',
    '%A': 'Weekday'
}

CFD_features =[
    'Race', 'Gender', 'Age',
    'Afraid', 'Angry', 'Attractive', 'Babyface', 'Disgusted', 'Dominant', 'Feminine',
    'Happy', 'Masculine', 'Prototypic', 'Sad', 'Suitability', 'Surprised', 'Threatening',
    'Trustworthy','Unusual',
    # 'NumberofRaters',
    # 'Female_prop','Male_prop',
    # 'Asian_prop', 'Black_prop', 'Latino_prop', 'Multi_prop','Other_prop', 'White_prop',
]

def gen_body():
    body = [
        # Activity Summary
        dcc.Markdown(""" 
    ## Activity Summary
    """),
        html.Div([
            html.Div([
                dcc.Markdown('Date-time bins: '),
                dcc.Checklist(
                    id='date-type',
                    options=[
                        {'label': 'Date', 'value': '%D'},
                        {'label': 'Hour', 'value': '%H'},
                        {'label': 'Weekday', 'value': '%A'},
                    ],
                    labelStyle={'display': 'inline-block'},
                    value=['%D'],
                ),
                dcc.Markdown('Colored by: '),
                dcc.Dropdown(
                    id='bar-color',
                    options=[{'label': label, 'value': value}
                             for (label, value) in main_vars.items()],
                    value='all',
                ),
            ], className='two columns'),

            # activity versus time
            html.Div([
                dcc.Graph(id='activity-vs-time'),
            ], className='four columns'),

            # activity for each database
            html.Div([
                dcc.Graph(id='activity-vs-DB'),
            ], className='four columns'),
        ], className='row'),
        html.Hr(),

        # Labels
        dcc.Markdown("""
    ## Labels vs. CFD 
        """),
        html.Div([
            # histogram for each user
            html.Div([
                dcc.Markdown('Select Users '),
                dcc.Dropdown(
                    id='user-uinfo',
                    options=[{'label': i, 'value': i} for i in LABELS[UINFO].unique()],
                    value=[],
                    multi=True
                ),
                dcc.Markdown('Database name: '),
                dcc.Dropdown(
                    id='db-name',
                    options=[{'label': i, 'value': i} for i in LABELS.db_name.unique()],
                    value='attractiveness_label',
                ),
                dcc.Markdown('CFD dataset: '),
                dcc.Dropdown(
                    id='scatter-cfd-name',
                    options=[{'label': i, 'value': i} for i in CFD_features],
                    value='Attractive',
                ),
                dcc.Markdown('Histnorm: '),
                dcc.RadioItems(
                    id='radio-histnorm',
                    options=[{'label': 'Count', 'value': 'None'},
                             {'label': 'Percent', 'value': 'percent'}],
                    labelStyle={'display': 'inline-block'},
                    value='percent',
                ),
            ], className='two columns'),

            # scatter plot users vs CFD features
            html.Div([
                dcc.Graph(id='scatter-user-cfd'),
            ], className='four columns'),
            # scatter plot users vs CFD features
            html.Div([
                dcc.Graph(id='user-label-hist'),
            ], className='four columns'),

        ], className='row'),

        # Selected Image
        dcc.Markdown(""" 
        ### Selected Image 
        """),
        html.Div([
            # dcc.Markdown(""" **Selected Image** """, id='md-test'),
            dcc.Graph(id='image', className='three columns'),
            dcc.Graph(id='image-radar', className='four columns'),
            dcc.Graph(id='image-info', className='three columns'),
        ], className='row'),

        # Compare
        html.Hr(),
        dcc.Markdown("""
    ## Compare Users 
        """),
        # html.Div([
        html.Div([
            dcc.Markdown(' #### User 1 '),
            dcc.Dropdown(
                id='u2u-uinfo-1',
                options=[{'label': i, 'value': i} for i in LABELS[UINFO].unique()],
                value=LABELS[UINFO].iloc[0],
            ),
            dcc.Markdown(' #### User 2 '),
            dcc.Dropdown(
                id='u2u-uinfo-2',
                options=[{'label': i, 'value': i} for i in LABELS[UINFO].unique()],
                value=LABELS[UINFO].iloc[-1],
            ),
        ], className='two columns'),
        dcc.Graph(id='user-vs-user', className='four columns'),
        dcc.Graph(id='user-vs-user-MI',className='four columns'),
        html.Div([
            dcc.Dropdown(
                id='u2u-db-name-1',
                options=[{'label': i, 'value': i} for i in LABELS.db_name.unique()],
                # value='attractiveness_label',
                value = None,
            ),
            dcc.Dropdown(
                id='u2u-db-name-2',
                options=[{'label': i, 'value': i} for i in LABELS.db_name.unique()],
                # value='attractiveness_label',
                value = None,
            ),
        ], className='two columns'),
        dcc.Graph(id='user-vs-cfd', className='four columns'),
        # ], className='row'),
    ]
    return body


app.layout = html.Div([
    # header
    dcc.Markdown(' ### Global settings: '),
    html.Button(id='refresh-button', n_clicks=0, children='Refresh'),
    html.Div('Threshold For User Activity'),
    dcc.Input('user-activity-threshold', type='number', placeholder='1000'),
    dcc.Markdown('Show User (not working now!) '),
    dcc.RadioItems( id='radio-uinfo',
        options=[
            {'label': 'Email', 'value': UINFO},
            {'label': 'Name', 'value': 'name'},
            {'label': 'User ID', 'value': 'user_id'}
        ],
        labelStyle={'display': 'inline-block'},
        value=UINFO,
    ),
    html.Div(gen_body(), id='div-body'),

    # hidden signal value
    html.Div(id='signal', style={'display': 'none'}),
    html.Div(id='signal-image', style={'display': 'none'}),
])

#
# @app.callback(Output('div-body', 'children'),
#               Input('refresh-button', 'n_clicks'),
#               Input('radio-uinfo', 'value'),
#               Input('user-activity-threshold', 'value'))
# def change_unifo(n_clicks, info, threshold):
#     global UINFO
#     global LABELS
#     UINFO = info
#     LABELS = get_labels(threshold)
#     return gen_body()


@app.callback(Output('user-vs-cfd', 'figure'),
              Input('u2u-db-name-1', 'value'),
              Input('u2u-db-name-2', 'value'))
def update_div(db_name1, db_name2):
    all_emails = LABELS.email.unique()
    MIs = np.zeros((len(all_emails),len(all_emails)))
    for i,email1 in enumerate(all_emails):
        for j, email2 in enumerate(all_emails):
            dff = LABELS
            dff1 = dff[(dff[UINFO] == email1)]
            dff2 = dff[(dff[UINFO] == email2)]
            if db_name1:
                dff1 = dff1[(dff1.db_name == db_name1)]
            if db_name2:
                dff2 = dff2[(dff2.db_name == db_name2)]
            dff = pd.merge(dff1,dff2,on='item_id', how='left')
            try:
                mi = MI(dff[['label_x']], dff[['label_y']])
            except Exception:
                mi = 0
            MIs[i,j] = mi
            if email1==email2:
                MIs[i,j] = 0
    # if db_name1==db_name2:
    #     for i in len(all_emails):
    #         MIs[i,i] = float('nan')
    all_emails = [e[:8]+'***' for e in all_emails]
    fig = px.imshow(MIs, x=all_emails, y=all_emails)
    fig.update_xaxes(side='top')
    return fig


@app.callback(Output('user-vs-user-MI', 'figure'),
              Input('u2u-uinfo-1', 'value'),
              Input('u2u-uinfo-2', 'value'))
def update_div(email1, email2):
    all_db_names = LABELS.db_name.unique()
    MIs = np.zeros((len(all_db_names),len(all_db_names)))
    for i,db_name1 in enumerate(all_db_names):
        for j, db_name2 in enumerate(all_db_names):
            dff = LABELS
            if db_name1:
                dff1 = dff[(dff[UINFO] == email1) & (dff.db_name == db_name1)]
            if db_name2:
                dff2 = dff[(dff[UINFO] == email2) & (dff.db_name == db_name2)]
            dff = pd.merge(dff1,dff2,on='item_id', how='left')
            try:
                mi = MI(dff[['label_x']], dff[['label_y']])
            except Exception:
                mi = 0
            MIs[i,j] = mi
    if email1==email2:
        for i in range(len(all_db_names)):
            MIs[i,i] = float('nan')
    fig = px.imshow(MIs, x=all_db_names, y=all_db_names)
    fig.update_xaxes(side='top')
    return fig


# @app.callback(Output('user-vs-cfd', 'figure'),
#               Input('u2u-uinfo-1', 'value'),
#               Input('u2u-db-name-1', 'value'),
#               Input('u2cfd', 'value'))
# def plot_u2u(email1, db_name, cfd_feature):
#     dff = LABELS[LABELS.db_name == db_name]
#     dff = dff[(dff[UINFO] == email1) ]
#     fig = px.density_heatmap(dff, x='label', y=cfd_feature,
#                              marginal_x='histogram', marginal_y='histogram')
#     return fig


@app.callback(Output('user-vs-user', 'figure'),
              Input('u2u-uinfo-1', 'value'),
              Input('u2u-uinfo-2', 'value'),
              Input('u2u-db-name-1', 'value'),
              Input('u2u-db-name-2', 'value'))
def plot_u2u(email1, email2, db_name, db_name2):
    # dff = LABELS[LABELS.db_name == db_name]
    dff = LABELS
    dff1 = dff[(dff[UINFO] == email1)]
    dff2 = dff[(dff[UINFO] == email2) ]
    if db_name:
        dff1 = dff[(dff[UINFO] == email1) & (dff.db_name == db_name)]
    if db_name2:
        dff2 = dff[(dff[UINFO] == email2) & (dff.db_name == db_name2)]
    dff = pd.merge(dff1,dff2,on='item_id', how='left')
    mutual_info = MI(dff[['label_x']], dff[['label_y']])
    fig = px.density_heatmap(dff, x='label_x', y='label_y',
                             marginal_x='histogram', marginal_y='histogram',
                             title='Mutual Information: {:2.2}'.format(mutual_info[0]))
    return fig


@app.callback(
    Output('image-info', 'figure'),
    Input('signal-image', 'children'),
    Input('user-uinfo', 'value')
)
def display_img_stats(img_info, uinfos):
    img_info = json.loads(img_info)
    dff = LABELS[LABELS.item_id == img_info['item_id']]
    # dff = dff[dff[UINFO].isin(uinfos)]
    dff.loc[dff.db_name == 'age_label', 'label'] /= 10
    dff.db_name = [db.replace('_', ' ') for db in dff.db_name]

    fig = go.FigureWidget(
        data=[go.Bar(x=dff[dff.db_name==db].label,
                     y=dff[dff.db_name==db].db_name,
                     orientation='h',
                     name = db,
                     marker={'opacity': 1.1/sum(dff.db_name==db),})
              for db in dff.db_name.unique()])
    fig.update_layout(barmode='overlay', xaxis={'showgrid': False})
    return fig

@app.callback(
    Output('image-radar', 'figure'),
    Input('signal-image', 'children'),
    Input('user-uinfo', 'value'),
)
def display_img_stats(img_info, uinfos):
    img_info = json.loads(img_info)
    dff = LABELS[LABELS.item_id == img_info['item_id']]
    if uinfos:
        dff = dff[dff[UINFO].isin(uinfos)]
    dff.loc[dff.db_name == 'age_label', 'label'] /= 10
    fig2 = px.line_polar(dff, r='label', theta='db_name', color=UINFO,
                         line_close=True,
                         # color_discrete_sequence=px.colors.sequential.Viridis,
                         # template='plotly_dark'
                         )
    # fig2.update_layout(showlegend=False)
    return fig2


@app.callback(
    Output('image', 'figure'),
    Input('signal-image','children')
)
def display_image(img_info):
    img_info = json.loads(img_info)
    img = Image.open(img_info['item_path'])
    fig = px.imshow(img)
    return fig




@app.callback(
    Output('signal-image', 'children'),
    Input('scatter-user-cfd', 'clickData'))
def get_image_path(clickData):
    if clickData:
        img_path = clickData['points'][0]['customdata'][0]
        img_id = clickData['points'][0]['customdata'][1]
    else:
        # img_id = 1
        img_path = LABELS.item_path.iloc[0]
        img_id = int(LABELS.item_id.iloc[0])
    img_path = utils.get_img_dir(img_path)
    return json.dumps({'item_path': img_path, 'item_id': img_id})


@app.callback(
    Output('scatter-user-cfd', 'figure'),
    Input('user-uinfo', 'value'),
    Input('db-name', 'value'),
    Input('scatter-cfd-name', 'value'),
)
def update_scatter(uinfos, db_name, cfd_name):
    dff = LABELS[LABELS.db_name == db_name]
    if len(uinfos) > 0:
        dff = dff[dff[UINFO].isin(uinfos)]
    fig = px.strip(dff, x='label', y=cfd_name, custom_data=['item_path','item_id'], color=UINFO)
    # fig.update_layout(showlegend=False)
    return fig


@app.callback(
    Output('user-label-hist', 'figure'),
    Input('user-uinfo', 'value'),
    Input('radio-histnorm', 'value'),
    Input('db-name', 'value'))
def update_hist_fig(uinfos, histnorm, db_name):
    if histnorm == 'None':
        histnorm = None
    dff = LABELS[LABELS.db_name == db_name]
    if len(uinfos) > 0:
        dff = dff[dff[UINFO].isin(uinfos)]
    fig = px.histogram(dff,
                       x='label',
                       color=UINFO,
                       barmode='group',
                       histnorm=histnorm, )
    if histnorm:
        fig.update_layout(yaxis_title=histnorm)
    return fig


@app.callback(
    Output('activity-vs-time', 'figure'),
    Input('bar-color', 'value'),
    Input('signal', 'children'),
)
def update_figure(selected_color, date_type):
    if selected_color == 'all':
        fig = px.histogram(LABELS, x="date_time")
    else:
        fig = px.histogram(LABELS, x="date_time", color=selected_color)
    fig.update_layout(xaxis_title=' '.join([date_time_dict[d] for d in date_type.split('-')]))
    return fig


@app.callback(
    Output('signal', 'children'),
    Input('date-type', 'value'))
def update_date(date_type):
    if date_type:
        date_type = '-'.join(date_type)
    else:
        date_type = '%D'
    LABELS['date_time'] = pd.to_datetime(LABELS['created_on'])
    LABELS['date_time'] = LABELS['date_time'].apply(lambda x: x.strftime(date_type))
    return date_type



@app.callback(
    Output('activity-vs-DB', 'figure'),
    Input('bar-color', 'value'))
def update_figure2(selected_color):
    if selected_color == 'all':
        fig = px.histogram(LABELS, y="db_name")
    else:
        fig = px.histogram(LABELS, y="db_name", color=selected_color)
    fig.update_layout(yaxis_title="DB name")
    return fig


if __name__ == '__main__':
    if is_running_local:
        app.run_server(debug=True)
    else:
        app.run_server(debug=True, host='0.0.0.0')
