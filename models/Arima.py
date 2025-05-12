import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
#from pmdarima import auto_arima


from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_arima(series, steps=24, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24)):
    model = SARIMAX(series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    fitted = model.fit(disp=False)
    forecast_obj = fitted.get_forecast(steps=steps)

    mean = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()
    std = (conf_int[:, 1] - conf_int[:, 0]) / 4

    return mean, std


def forecast_prophet(series, steps=24, freq='H'):
    if isinstance(series, np.ndarray):
        raise ValueError("Prophet needs a pandas Series with datetime index, not a numpy array.")

    df = pd.DataFrame({'ds': series.index, 'y': series.values})
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=steps, freq=freq)
    forecast = model.predict(future)

    mean_pred = forecast['yhat'][-steps:].to_numpy()
    std_pred = ((forecast['yhat_upper'] - forecast['yhat_lower']) / 4)[-steps:].to_numpy()
    return mean_pred, std_pred

