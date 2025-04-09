import pandas as pd
import numpy as np
np.random.seed(1)

import statsmodels as st
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.seasonal import seasonal_decompose

import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots

import sklearn.metrics as metrics

import os
os.environ['NIXTLA_ID_AS_COL'] = '1'

from pytorch_lightning import Trainer
trainer = Trainer(
    max_steps=4,
    logger=False,
    enable_progress_bar=False,
    enable_model_summary=False  # Disable model summary
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")




# DATA =============================================================================================
ponte = pd.read_pickle(r'Data\Data_Ponte_dos_Remedios.pkl')
del ponte['o3']
guarulhos = pd.read_pickle(r'Data\Data_Guarulhos.pkl')
guarulhos = guarulhos[['date','o3']]
data = ponte.merge(guarulhos, on='date', how='outer')
data.reset_index(drop=True)
# ====================================================================================================






# TIME OBJECT =======================================================================================
class TimeObject:
    def __init__(self, df:pd.DataFrame, column:str, agg_freq:str='W',
                 col_name:str='AQI', time_col_name:str='date',
                 NAN_treatment_args:dict={'method':'from_derivatives'}) -> None:

        self.df = df[[time_col_name,column]]
        self.column = column
        self.col_name = col_name
        self.time_col_name = time_col_name
        self.time_serie = self.to_serie_()
        self.NAN_treatment_args = NAN_treatment_args

        if self.NAN_treatment_args is not None: self.NAN_treatment_(**self.NAN_treatment_args)
        self.NIXTLA_treatment_()
        if agg_freq != None: 
            self.nixtla_df = self.nixtla_df.groupby(pd.Grouper(key='ds', freq=agg_freq)).agg({'y': 'mean'}).reset_index()
            self.nixtla_df.loc[:, ['unique_id']] = 1.0
        self.NIXTLA_train_test(split=7)
# ====================================================================================================
    def to_serie_(self) -> pd.Series:
        time_serie = self.df[self.column].fillna(np.nan)
        time_serie.index = pd.to_datetime(self.df[self.time_col_name])

        full_index = pd.date_range(start=time_serie.index.min(), end=time_serie.index.max(), freq='D')
        time_serie = time_serie.reindex(full_index)
        return time_serie
# ====================================================================================================
    def NAN_treatment_(self, **kwargs) -> None:
        self.time_serie = self.time_serie.interpolate(**kwargs)
# ====================================================================================================
    def NIXTLA_treatment_(self) -> None:
        self.nixtla_df = pd.DataFrame()
        self.nixtla_df.loc[:, ['ds']] = pd.to_datetime(self.time_serie.index)
        self.nixtla_df.loc[:, ['y']] = self.time_serie.values
        self.nixtla_df.loc[:, ['unique_id']] = 1.0
# ====================================================================================================
    def plot(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(trace=go.Scatter(
            x=self.time_serie.index, y=self.time_serie,
            marker=dict(color='#222222')
        ))
        return fig
# ====================================================================================================
    def NIXTLA_train_test(self, split:int=12):
        self.split = split
        self.Y_train = self.nixtla_df[self.nixtla_df.ds<self.nixtla_df['ds'].values[-split]]
        self.Y_test = self.nixtla_df[self.nixtla_df.ds>=self.nixtla_df['ds'].values[-split]].reset_index(drop=True)
# ====================================================================================================
    def metrics_(self, forecast_df:pd.DataFrame, method:str='NHITS'):

        def smape(y_true, y_pred):
            summation = 0
            for i in range(len(y_true)):
                summation += np.abs(y_true[i]-y_pred[i])/(np.abs(y_true[i]) + np.abs(y_pred[i]))
            return 200/(len(y_true)+1) * summation
        
        self.metrics = {}
        self.metrics['mae'] = np.round(metrics.mean_absolute_error(y_true=self.Y_test['y'], y_pred=forecast_df[method]),5)
        self.metrics['mape'] = np.round(100*metrics.mean_absolute_percentage_error(y_true=self.Y_test['y'], y_pred=forecast_df[method]),5)
        self.metrics['mse'] = np.round(metrics.mean_squared_error(y_true=self.Y_test['y'], y_pred=forecast_df[method]),5)
        self.metrics['max'] = np.round(metrics.max_error(y_true=self.Y_test['y'], y_pred=forecast_df[method]),5)
        self.metrics['smape'] = np.round(smape(y_true=self.Y_test['y'], y_pred=forecast_df[method]),5)
        return
# ====================================================================================================
    def plot_time_series(self):
        fig = go.Figure()
        fig.add_trace(trace=go.Scatter(
            x=self.nixtla_df['ds'], y=self.nixtla_df['y'],
            mode='lines', marker=go.scatter.Marker(
                color='black'
            ), name='Time Series'
        ))
        main_layout(fig=fig, width=1100, height=450, title='Time Series', x='time', y=self.col_name)
        return fig
# ====================================================================================================
    def plot_forecast(self, forecast_df:pd.DataFrame, confidence:int=90, method='NHITS', show:bool=True, show_metrics:bool=True):
        fig = go.Figure()

        fig.add_trace(trace=go.Scatter(
            x=self.Y_train['ds'], y=self.Y_train['y'],
            mode='lines', marker=go.scatter.Marker(
                color='black'
            ), name='train'
        ))
        
        fig.add_trace(trace=go.Scatter(
            x=self.Y_test['ds'], y=self.Y_test['y'],
            mode='lines', marker=go.scatter.Marker(
                color='skyblue'
            ), name='test'
        ))

        fig.add_trace(trace=go.Scatter(
            x=forecast_df['ds'], y=forecast_df[f'{method}'],
            mode='lines', marker=go.scatter.Marker(
                color='orange'
            ), name=method
        ))

        try:
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'], y=forecast_df[f'{method}-lo-{confidence}'],
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0)',
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=forecast_df['ds'], y=forecast_df[f'{method}-hi-{confidence}'],
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0.2)',
                name=f'confidence: {confidence}%'
            ))
        except: ...

        main_layout(fig=fig, width=1100, height=450, title=f'Forecast - {self.column.upper()} - Horizon: {int(len(self.Y_test))}', x='time', y=self.col_name)

        if show:
            fig.show()
        if show_metrics:
            self.metrics_(forecast_df, method=method)
            for key, metric in self.metrics.items():
                print(f'{key}: {metric}')
        
        return fig
# ====================================================================================================
    def plot_seasonal_decompose(self, width=1000, height=700, period=52, x_range=[0,353], model='add'):
        decomposed=seasonal_decompose(self.nixtla_df["y"], model=model, period=period)
        self.trend = decomposed.trend
        self.seasonal = decomposed.seasonal
        self.resid = decomposed.resid

        fig = plotly.subplots.make_subplots(rows=4, cols=1, row_titles=['','Trend','Seasonality','Residual'])

        fig.add_trace(go.Scatter(
            x=[i for i in range(len(decomposed.trend))], y=self.nixtla_df['y'],
            mode='lines', marker=dict(color='black'), showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[i for i in range(len(decomposed.trend))], y=decomposed.trend,
            mode='lines', marker=dict(color='black'), showlegend=False
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=[i for i in range(len(decomposed.seasonal))], y=decomposed.seasonal,
            mode='lines', marker=dict(color='black'), showlegend=False
        ), row=3, col=1)

        if model == 'add':
            fig.add_trace(go.Scatter(
                x=[i for i in range(len(decomposed.resid))], y=[0 for i in range(len(decomposed.resid))],
                mode='lines', line=dict(color='red', dash='dash'), showlegend=False
            ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=[i for i in range(len(decomposed.resid))], y=decomposed.resid,
            mode='markers', marker=dict(color='black'), showlegend=False
        ), row=4, col=1)

        main_subplot_layout(fig, width=width, height=height, title=f'Seasonal Decomposition - {self.column.upper()}', x_range=x_range)
        
        return fig
# ====================================================================================================
    def plot_acf_pacf(self, function='pacf', nlags=100, width=1000, height=500):
        corr_array = pacf(self.nixtla_df['y'].dropna(), alpha=0.05, nlags=nlags) if (function=='pacf') else acf(self.nixtla_df['y'].dropna(), alpha=0.05, nlags=nlags)
        lower_y = corr_array[1][:,0] - corr_array[0]
        upper_y = corr_array[1][:,1] - corr_array[0]

        fig = go.Figure()
        [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#222222') 
        for x in range(len(corr_array[0]))]
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='rgba(255, 150, 100, 1)',
                    marker_size=7)
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines', fillcolor='rgba(255, 100, 100, 0.1)',
                fill='tonexty', line_color='rgba(255,255,255,0)')
        
        fig.update_traces(showlegend=False)
        title=f'Partial Autocorrelation Function (PACF) - {self.column.upper()}' if (function=='pacf') else f'Autocorrelation Function (ACF) - {self.column.upper()}'
        main_layout(fig, title=title, x_range=[-1,nlags+1], width=width, height=height)
        fig.update_yaxes(zerolinecolor='#000000')
        
        return fig
# ====================================================================================================
    def plot_boxplot(self, width=1400, height=400):
        fig = go.Figure()

        fig.add_trace(go.Box(
            x=self.nixtla_df['y'], boxmean=True, name='',
            marker=dict(color='black'), showlegend=False, jitter=0.3,
            boxpoints='all'
        ))

        main_layout(fig, width=width, height=height, title=f'Dispersion')

        return fig
# ====================================================================================================
    def plot_train_test_interval(self, initial_train_date:str=None, upto_train_date:str=None, 
                                 initial_test_date:str=None, upto_test_date:str=None, index_plot:list=None,
                                 train_color:str='108, 99, 96', test_color:str='167, 73, 193',
                                 show_entire_time_series:bool=False, min_y:int=0, max_y:int=100,
                                 width:int=1100, height:int=450):
        """
        ``index_plot``: [First_Train_Obs, Last_Train_Obs, Horizon]
        """
        fig = go.Figure()
        try:
            if index_plot is not None:
                initial_train_date = self.nixtla_df['ds'].dt.strftime('%Y-%m-%d').iloc[index_plot[0]]
                upto_train_date = self.nixtla_df['ds'].dt.strftime('%Y-%m-%d').iloc[index_plot[0]+index_plot[1]-1]
                upto_test_date = self.nixtla_df['ds'].dt.strftime('%Y-%m-%d').iloc[index_plot[0]+index_plot[1]-1+index_plot[2]]
        except IndexError:
            raise IndexError(
                f'IndexError | Only {len(self.nixtla_df)} rows available, '
                f'but tried to access index {index_plot[1] + index_plot[2]}'
            )
        if initial_test_date is None: initial_test_date = upto_train_date
        fig.add_trace(go.Scatter(
            x=[initial_train_date,initial_train_date], y=[min_y, max_y], showlegend=False,
            mode='lines', line=dict(color=f'rgb({train_color})', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[upto_train_date,upto_train_date], y=[min_y, max_y], name='Train Window',
            mode='lines', line=dict(color=f'rgb({train_color})', dash='dash'),
            fill='tonexty', fillcolor=f'rgba({train_color}, 0.15)'
        ))
        fig.add_trace(go.Scatter(
            x=[initial_test_date,initial_test_date], y=[min_y, max_y], showlegend=False,
            mode='lines', line=dict(color=f'rgb({test_color})', dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=[upto_test_date,upto_test_date], y=[min_y, max_y], name='Test Window',
            mode='lines', line=dict(color=f'rgb({test_color})', dash='dash'),
            fill='tonexty', fillcolor=f'rgba({test_color}, 0.1)'
        ))
        if show_entire_time_series:
            fig.add_trace(trace=go.Scatter(
                x=self.nixtla_df['ds'], y=self.nixtla_df['y'],
                mode='lines', marker=go.scatter.Marker(
                    color='black'
                ), name='Time Series'
            ))
        else:
            fig.add_trace(trace=go.Scatter(
                x=self.nixtla_df[self.nixtla_df['ds'] <= self.Y_test['ds'].dt.strftime('%Y-%m-%d').values[-1]]['ds'],
                y=self.nixtla_df[self.nixtla_df['ds'] <= self.Y_test['ds'].dt.strftime('%Y-%m-%d').values[-1]]['y'],
                mode='lines', marker=go.scatter.Marker(
                    color='black'
                ), name='Time Series'
            ))
        main_layout(fig=fig, width=width, height=height, title='Time Series | Train & Test Windows', x='time', y=self.col_name, y_range=[0,None])
        return fig
# ====================================================================================================
    def fixed_origin_rolling_window_cross_validation(self, initial_train_date=None, 
                                                     total_data_points:int=100, h=7,
                                                     plot_interval:bool=True, **kwargs):
        
        if (initial_train_date is not None) & (isinstance(initial_train_date, str)):
            self.nixtla_df = self.nixtla_df[self.nixtla_df['ds'] >= initial_train_date]
        if (initial_train_date is not None) & (isinstance(initial_train_date, int)):
            self.nixtla_df = self.nixtla_df.iloc[initial_train_date:,:]

        self.Y_train = self.nixtla_df.iloc[:total_data_points-h,:]
        self.Y_test = self.nixtla_df.iloc[total_data_points-h:total_data_points,:]

        if plot_interval:
            self.plot_train_test_interval(
                initial_train_date=self.Y_train['ds'].dt.strftime('%Y-%m-%d').values[0],
                upto_train_date=self.Y_train['ds'].dt.strftime('%Y-%m-%d').values[-1],
                initial_test_date=self.Y_train['ds'].dt.strftime('%Y-%m-%d').values[-1],
                upto_test_date=self.Y_test['ds'].dt.strftime('%Y-%m-%d').values[-1],
                **kwargs
            ).show()

        return self.Y_test







# USED WITH PLOTLY TO ELABORATE THE DESIGN ===========================================================
def main_layout(fig:go.Figure, width=700, height=600, x=None, y=None, title=None,
               x_range=None, y_range=None, paper_color='white', 
               customdata=None, hover_customdata='Info', 
               hover_x='x',hover_y='y', **kwargs) -> go.Figure:
    fig.layout = go.Layout(
        width=width,
        height=height,
        plot_bgcolor=paper_color,
        paper_bgcolor=paper_color,
        xaxis={'gridcolor':'#cccccc', 'linecolor':'black','title':x, 'range':x_range},
        yaxis={'gridcolor':'#cccccc', 'linecolor':'black','title':y, 'range':y_range},
        title={'text':title},
        **kwargs
    )
    if customdata == 'no':
        ...
    elif customdata is None:
        fig.update_traces(patch={
            'customdata':customdata, 'hovertemplate': hover_x + ': %{x}<br>' + hover_y + ': %{y}'
        })
    else:
        fig.update_traces(patch={
            'customdata':customdata,
            'hovertemplate': hover_x + ': %{x}<br>' + hover_y + ': %{y}<br>' + hover_customdata + ': %{customdata}<br>'
        })
    return fig
# ====================================================================================================
def main_subplot_layout(fig:go.Figure, width=1400, height=500, title=None, paper_color='white',
                        x=None, y=None, rows=1, cols=2, x_range=None, y_range=None,
                        customdata=None, hover_customdata='Info', 
                        hover_x='x',hover_y='y', **kwargs) -> go.Figure:
    fig.update_layout({
        'width':width,
        'height':height,
        'plot_bgcolor':paper_color,
        'paper_bgcolor':paper_color,
        'title':title,
        **kwargs
    })
    for xaxis in fig.select_xaxes():
        xaxis.update(
            showgrid=True,
            gridcolor='#CCCCCC',
            linecolor='black',
            title=x,
            range=x_range
        )
    for yaxis in fig.select_yaxes():
        yaxis.update(
            showgrid=True,
            gridcolor='#CCCCCC',
            linecolor='black',
            title=y,
            range=y_range
        )
    if customdata == 'no':
        ...
    elif customdata is None:
        fig.update_traces(patch={
            'customdata':customdata, 'hovertemplate': hover_x + ': %{x}<br>' + hover_y + ': %{y}'
        })
    else:
        fig.update_traces(patch={
            'customdata':customdata,
            'hovertemplate': hover_x + ': %{x}<br>' + hover_y + ': %{y}<br>' + hover_customdata + ': %{customdata}<br>'
        })
    return fig
# ====================================================================================================