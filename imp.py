import pandas as pd
from copy import deepcopy
from datetime import datetime, timedelta, date
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
# from darts.models.forecasting.prophet_model import Prophet
from darts import TimeSeries
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm
import logging
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.auto_arima import AutoARIMA
from darts.models.forecasting.theta import Theta
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.sf_auto_arima import StatsForecastAutoARIMA
from fbprophet import Prophet
