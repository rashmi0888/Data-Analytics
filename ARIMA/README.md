1. There is 1 python file in this folder - ARIMA.py
2. There is 1 test file in this folder - River_Flow.csv. It is on this file that ARIMA analysis is performed.
3. There is 1 jupyter notebook - ARIMA.ipynb, which runs directly on browser.
4. This code does the AR. MA part is skipped as it is outside the scope of project.
5. The file could be run as:  python ARIMA.py
6. Apart from plots, it also outputs various values on console.



Functions | Input | Output          
------------ | --------------- | ------------------
calculate_acf() | Timeseries, no of lags | ACF plot
calculate_pacf() | Timeseries, no of lags | PACF plot
test_stationarity() | Timeseries | Whether timeseries is stationary?
log_transform(), diff() | Timeseries | Stationary timeseries
model_identification() | Prompts user to enter value for p in AR(p) model by looking at PACF plot | p
autoregression_coefficients() | Timeseries, no of lags | Coefficients, estimated values as per AR(p) model
forecast() | | Does forecasting for AR(p) model
main() | | 1.Plots original time series 2. Plots ACF and PACF for both non-stationary and stationary time series 3.Plots AR(p) model fit along with forecasted values
