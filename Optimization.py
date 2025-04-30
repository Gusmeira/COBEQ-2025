import pandas as pd
import numpy as np
np.random.seed(1)
import optuna
import itertools
import shutil
from functools import partial

import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots
import time

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
import sklearn.metrics as metrics
import scipy.stats as stats

from statsforecast import StatsForecast
from statsforecast.models import AutoCES, AutoETS, AutoTheta

ponte = pd.read_pickle(r'Data\Data_Ponte_dos_Remedios.pkl')
del ponte['o3']
guarulhos = pd.read_pickle(r'Data\Data_Guarulhos.pkl')
guarulhos = guarulhos[['date','o3']]

data = ponte.merge(guarulhos, on='date', how='outer')
data.reset_index(drop=True)

import joblib
import pickle
from IPython.display import clear_output
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

from TimeObjectModule import TimeObject







# ==========================================================================================================
# CLEAR FUNCTION ===========================================================================================
def clear_terminal():
    # Check the operating system
    if os.name == 'nt':  # For Windows
        os.system('cls')
    else:  # For macOS and Linux
        os.system('clear')
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================






# ==========================================================================================================
# NHITS ====================================================================================================
# ==========================================================================================================
def objective_nhits(trial, pollutant, horizon):
    # Hyperparameter search space
    input_size = trial.suggest_int('input_size', 90, 1100, step=1)
    n_stacks = trial.suggest_int('n_stacks', 3, 7, step=1)
    n_blocks = trial.suggest_int('n_blocks', 1, 7, step=1)
    max_steps = trial.suggest_int('max_steps', 10, 700, step=1)
    local_scalar_type = trial.suggest_categorical('local_scalar_type', [None, 'standard', 'boxcox', 'minmax'])
    n_pool_kernel_size = trial.suggest_categorical('n_pool_kernel_size', [list(combination) for combination in list(itertools.product([1, 2, 3], repeat=3))])
    n_freq_downsample = trial.suggest_categorical('n_freq_downsample', [list(combination) for combination in list(itertools.product([1, 7, 90, 180, 365], repeat=3))])

    mape = []
    smape = []
    max = []
    mae = []
    mse = []
    # Split for cross validation
    max_k_fold = 4
    for k_fold in range(0,max_k_fold):
        print(f'\nPollutant = {pollutant} \nh = {horizon} \nTrial = {trial.number+1}\nFold = {k_fold+1}\n')
        # Instantiate TimeObject and prepare training data
        obj = TimeObject(df=data, column=pollutant, agg_freq='D')
        obj.fixed_origin_rolling_window_cross_validation(
            initial_train_date=None,
            total_data_points=(len(obj.nixtla_df)-(max_k_fold*horizon))-(k_fold*horizon),
            h=horizon,
            plot_interval=False
        )

        # Define the model
        model = NHITS(
            h=horizon,
            input_size=input_size,
            stack_types=n_stacks*['identity'],
            n_freq_downsample=n_freq_downsample+(n_stacks-len(n_freq_downsample))*[1],
            n_blocks=n_stacks*[n_blocks],
            n_pool_kernel_size=(n_stacks-len(n_pool_kernel_size))*[1]+n_pool_kernel_size,
            pooling_mode="MaxPool1d",
            activation="ReLU",
            interpolation_mode='linear',
            max_steps=max_steps,
            val_check_steps=10,
            early_stop_patience_steps=int(np.round(max_steps/(20),0)),
        )

        # Initialize NeuralForecast and fit the model
        fcst = NeuralForecast(
            models=[model],
            freq='D',
            local_scaler_type=local_scalar_type
        )
        fcst.fit(df=obj.Y_train, verbose=False, val_size=horizon+1)
        prediction = fcst.predict(df=obj.Y_train, verbose=False)

        # Evaluate metrics
        obj.metrics_(forecast_df=prediction, method='NHITS')
        mape.append(obj.metrics['mape'])
        smape.append(obj.metrics['smape'])
        max.append(obj.metrics['max'])
        mae.append(obj.metrics['mae'])
        mse.append(obj.metrics['mse'])
        
        try: clear_output(wait=True)
        except: ...
        try: clear_terminal()
        except: ...

    try:
        directory_path = "lightning_logs"
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
    except:
        ...

    mape = np.mean(mape)
    smape = np.mean(smape)
    max = np.mean(max)
    mae = np.mean(mae)
    mse = np.mean(mse)

    # Collect the results
    results.append({
        'pollutant': pollutant,
        'freq': 'D',
        'fold': max_k_fold,
        'trial': trial.number+1,
        'train_len': f'{round(100*len(obj.Y_train)/len(obj.nixtla_df),3)}',
        'h': horizon,
        'input_size': input_size,
        'n_stacks': n_stacks,
        'n_blocks': n_blocks,
        'max_steps': max_steps,
        'local_scalar_type': local_scalar_type,
        'n_pool_kernel_size': n_pool_kernel_size,
        'n_freq_downsample': n_freq_downsample,
        'mape': mape,
        'smape': smape,
        'max': max,
        'mae': mae,
        'mse': mse,
    })

    # The objective for Optuna is to minimize the MAE (or maximize a metric)
    return smape, mae  # Any metric you want to optimize

for pollutant in data[['pm10']]:
    for h in [30, 60, 90]:
        # Initialize the results list
        results = []
        # Define the optimization study_nhits
        study_nhits = optuna.create_study(directions=['minimize','minimize'])  # Minimize the MAE

        # Run the optimization with the number of trials you want
        study_nhits.optimize(partial(objective_nhits, pollutant=pollutant, horizon=h), n_trials=200)

        try: clear_output(wait=True)
        except: ...
        try: clear_terminal()
        except: ...

        NHITS_W = pd.DataFrame(results)

        output_dir = fr'Results COBEQ\NHITS (D)\{pollutant}'
        os.makedirs(output_dir, exist_ok=True)
        NHITS_W.to_pickle(fr'Results COBEQ\NHITS (D)\{pollutant}\{h}D_Df.pkl')
        joblib.dump(study_nhits, fr"Results COBEQ\NHITS (D)\{pollutant}\{h}D_Study.pkl")
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================



# ==========================================================================================================
# NBEATS ===================================================================================================
# ==========================================================================================================
def objective_nbeats(trial, pollutant, horizon):
    # Hyperparameter search space
    input_size = trial.suggest_int('input_size', 90, 1100, step=1)
    n_stacks = trial.suggest_int('n_stacks', 2, 7, step=1)
    n_blocks = trial.suggest_int('n_blocks', 1, 5, step=1)
    max_steps = trial.suggest_int('max_steps', 10, 700, step=1)
    local_scalar_type = trial.suggest_categorical('local_scalar_type', [None, 'standard', 'boxcox', 'minmax'])
    interpretability = trial.suggest_categorical('interpretability', [list(combination) for combination in list(itertools.product(['seasonality', 'trend', 'identity'], repeat=2))])

    mape = []
    smape = []
    max = []
    mae = []
    mse = []
    # Split for cross validation
    max_k_fold = 4
    for k_fold in range(0,max_k_fold):
        print(f'\nPollutant = {pollutant} \nh = {horizon} \nTrial = {trial.number+1}\nFold = {k_fold+1}\n')
        # Instantiate TimeObject and prepare training data
        obj = TimeObject(df=data, column=pollutant, agg_freq='D')
        obj.fixed_origin_rolling_window_cross_validation(
            initial_train_date=None,
            total_data_points=(len(obj.nixtla_df)-(max_k_fold*horizon))-(k_fold*horizon),
            h=horizon,
            plot_interval=False
        )

        # Define the model
        model = NBEATS(
            h=horizon,
            input_size=input_size,
            stack_types=interpretability+(n_stacks-len(interpretability))*['identity'],
            n_blocks=n_stacks * [n_blocks],
            max_steps=max_steps,
            learning_rate=1e-3,
            val_check_steps=10,
            early_stop_patience_steps=int(np.round(max_steps/(20),0)),
        )

        # Initialize NeuralForecast and fit the model
        fcst = NeuralForecast(
            models=[model],
            freq='D',
            local_scaler_type=local_scalar_type
        )
        fcst.fit(df=obj.Y_train, verbose=False, val_size=horizon+1)
        prediction = fcst.predict(df=obj.Y_train, verbose=False)

        # Evaluate metrics
        obj.metrics_(forecast_df=prediction, method='NBEATS')
        mape.append(obj.metrics['mape'])
        smape.append(obj.metrics['smape'])
        max.append(obj.metrics['max'])
        mae.append(obj.metrics['mae'])
        mse.append(obj.metrics['mse'])
            
        try: clear_output(wait=True)
        except: ...
        try: clear_terminal()
        except: ...

        try:
            directory_path = "lightning_logs"
            if os.path.exists(directory_path):
                shutil.rmtree(directory_path)
        except:
            ...

    mape = np.mean(mape)
    smape = np.mean(smape)
    max = np.mean(max)
    mae = np.mean(mae)
    mse = np.mean(mse)

    # Collect the results
    results.append({
        'pollutant': pollutant,
        'freq': 'D',
        'fold': max_k_fold,
        'trial': trial.number+1,
        'train_len': f'{round(100*len(obj.Y_train)/len(obj.nixtla_df),3)}',
        'h': horizon,
        'input_size': input_size,
        'n_stacks': n_stacks,
        'n_blocks': n_blocks,
        'max_steps': max_steps,
        'local_scalar_type': local_scalar_type,
        'interpretability': interpretability,
        'mape': mape,
        'smape': smape,
        'max': max,
        'mae': mae,
        'mse': mse,
    })

    # The objective for Optuna is to minimize the MAE (or maximize a metric)
    return smape, mae  # Any metric you want to optimize

for pollutant in data[['pm10']]:
    for h in [30, 60, 90]:
        # Initialize the results list
        results = []
        # Define the optimization study_nbeats
        study_nbeats = optuna.create_study(directions=['minimize','minimize'])  # Minimize the MAE

        # Run the optimization with the number of trials you want
        study_nbeats.optimize(partial(objective_nbeats, pollutant=pollutant, horizon=h), n_trials=200)

        try: clear_output(wait=True)
        except: ...
        try: clear_terminal()
        except: ...

        NBEATS_W = pd.DataFrame(results)

        output_dir = fr'Results COBEQ\NBEATS (D)\{pollutant}'
        os.makedirs(output_dir, exist_ok=True)
        NBEATS_W.to_pickle(fr'Results COBEQ\NBEATS (D)\{pollutant}\{h}D_Df.pkl')
        joblib.dump(study_nbeats, fr"Results COBEQ\NBEATS (D)\{pollutant}\{h}D_Study.pkl")
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================





    

# ==========================================================================================================
# STATS ====================================================================================================
# ==========================================================================================================
for pollutant in ['pm10',]:
    for h in [30, 60, 90]: #  30, 60, 90, 120
        results_stats = pd.DataFrame()

        mape = []
        smape = []
        max = []
        mae = []
        mse = []

        max_k_fold = 4
        for k_fold in range(0,max_k_fold):
            print(f'\nPollutant = {pollutant} \nh = {h}\nFold = {k_fold+1}\n')
            # Instantiate TimeObject and prepare training data
            obj = TimeObject(df=data, column=pollutant, agg_freq='D')
            obj.fixed_origin_rolling_window_cross_validation(
                initial_train_date=None,
                total_data_points=(len(obj.nixtla_df))-(k_fold*h),
                h=h,
                plot_interval=False
            )

            season_length = 365 # Monthly data 

            models = [
                # AutoARIMA(season_length=season_length, alias='AutoARIMA'),
                AutoCES(season_length=season_length, model='Z', alias='AutoCES-Z'),
                AutoCES(season_length=season_length, model='S', alias='AutoCES-S'),
                AutoCES(season_length=season_length, model='P', alias='AutoCES-P'),
                AutoCES(season_length=season_length, model='N', alias='AutoCES-N'),
                AutoTheta(season_length=season_length, decomposition_type="multiplicative", alias='AutoTheta-Multi'),
                AutoTheta(season_length=season_length, decomposition_type="additive", alias='AutoTheta-Add'),
            ]
            models = models + [
                AutoETS(season_length=season_length, model=ets, alias=f'AutoETS-{ets}')
                for ets in ['ZAZ', 'ZAN', 'ZAA', 'ZAM', 'ZNN']]

            frct = StatsForecast(models=models, freq='D')
            frct.fit(df=obj.Y_train)
            predicted = frct.predict(h=h)

            columns = predicted.columns
            columns = columns[(columns != 'ds') & (columns != 'unique_id')]

            for method in columns:
                obj.metrics_(predicted, method=method)
                results_stats = pd.concat([results_stats, pd.DataFrame({
                    'pollutant': [pollutant],
                    'method': [method],
                    'freq': ['D'],
                    'h': [h],
                    'mape': [obj.metrics['mape']],
                    'smape': [obj.metrics['smape']],
                    'max': [obj.metrics['max']],
                    'mae': [obj.metrics['mae']],
                    'mse': [obj.metrics['mse']]
                })])
            
            # ======================================================================================================

            nbeats = joblib.load(fr"Results COBEQ\NBEATS (D)\{pollutant}\{h}D_Study.pkl")
            model = NBEATS(
                h=h,
                input_size=nbeats.best_trials[0].params.get('input_size'),
                stack_types=nbeats.best_trials[0].params.get('interpretability')+(nbeats.best_trials[0].params.get('n_stacks')-len(nbeats.best_trials[0].params.get('interpretability')))*['identity'],
                n_blocks=nbeats.best_trials[0].params.get('n_stacks') * [nbeats.best_trials[0].params.get('n_blocks')],
                max_steps=nbeats.best_trials[0].params.get('max_steps'),
                learning_rate=1e-3,
                val_check_steps=10,
            )
            fcst = NeuralForecast(
                models=[model],
                freq='D',
                local_scaler_type=nbeats.best_trials[0].params.get('local_scalar_type')
            )
            fcst.fit(df=obj.Y_train, verbose=False)
            predicted = fcst.predict(df=obj.Y_train, verbose=False)
            obj.metrics_(predicted, method='NBEATS')
            results_stats = pd.concat([results_stats, pd.DataFrame({
                'pollutant': [pollutant],
                'method': ['NBEATS'],
                'freq': ['D'],
                'h': [h],
                'mape': [obj.metrics['mape']],
                'smape': [obj.metrics['smape']],
                'max': [obj.metrics['max']],
                'mae': [obj.metrics['mae']],
                'mse': [obj.metrics['mse']]
            })])

            # ======================================================================================================
            
            nhits = joblib.load(fr"Results COBEQ\NHITS (D)\{pollutant}\{h}D_Study.pkl")
            model = NHITS(
                h=h,
                input_size=nhits.best_trials[0].params.get('input_size'),
                stack_types=nhits.best_trials[0].params.get('n_stacks')*['identity'],
                n_freq_downsample=nhits.best_trials[0].params.get('n_freq_downsample')+(nhits.best_trials[0].params.get('n_stacks')-len(nhits.best_trials[0].params.get('n_freq_downsample')))*[1],
                n_blocks=nhits.best_trials[0].params.get('n_stacks')*[nhits.best_trials[0].params.get('n_blocks')],
                n_pool_kernel_size=(nhits.best_trials[0].params.get('n_stacks')-len(nhits.best_trials[0].params.get('n_pool_kernel_size')))*[1]+nhits.best_trials[0].params.get('n_pool_kernel_size'),
                pooling_mode="MaxPool1d",
                activation="ReLU",
                interpolation_mode='linear',
                max_steps=nhits.best_trials[0].params.get('max_steps'),
                val_check_steps=10,
            )
            fcst = NeuralForecast(
                models=[model],
                freq='D',
                local_scaler_type=nhits.best_trials[0].params.get('local_scalar_type')
            )
            fcst.fit(df=obj.Y_train, verbose=False)
            predicted = fcst.predict(df=obj.Y_train, verbose=False)
            obj.metrics_(predicted, method='NHITS')
            results_stats = pd.concat([results_stats, pd.DataFrame({
                'pollutant': [pollutant],
                'method': ['NHITS'],
                'freq': ['D'],
                'h': [h],
                'mape': [obj.metrics['mape']],
                'smape': [obj.metrics['smape']],
                'max': [obj.metrics['max']],
                'mae': [obj.metrics['mae']],
                'mse': [obj.metrics['mse']]
            })])
            
            try: clear_output(wait=True)
            except: ...
            try: clear_terminal()
            except: ...

        # ======================================================================================================

        results_stats = pd.DataFrame(results_stats)

        output_dir = fr'Results COBEQ\Stats (D)\{pollutant}'
        os.makedirs(output_dir, exist_ok=True)
        results_stats.to_pickle(fr'Results COBEQ\Stats (D)\{pollutant}\{h}D_Df.pkl')