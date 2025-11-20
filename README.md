# Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms

This project implements and evaluates a sophisticated deep learning model for time series forecasting, specifically focusing on incorporating attention mechanisms to enhance predictive accuracy and interpretability. The `AirPassengers` dataset is used, and the performance of an Encoder-Decoder LSTM model with self-attention is compared against a SARIMA benchmark model.

 1. Complete, Well-Documented Python Code Implementation

python
 ==================================================================================
  SECTION 1: Data Preparation
 ==================================================================================

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, concatenate, Dot, Activation
import seaborn as sns

 1.1 Load the AirPassengers dataset
 Using get_rdataset as it's more robust for this dataset
air_passengers_data = sm.datasets.get_rdataset("AirPassengers", "datasets").data

Convert 'time' column to proper datetime index
air_passengers_data['year'] = air_passengers_data['time'].apply(lambda x: int(x))
air_passengers_data['month'] = air_passengers_data['time'].apply(lambda x: int(round((x - int(x)) * 12)) + 1)
air_passengers_data['date'] = pd.to_datetime(air_passengers_data['year'].astype(str) + '-' + air_passengers_data['month'].astype(str))
df = air_passengers_data.set_index('date')['value'].to_frame()
df.rename(columns={'value': 'passengers'}, inplace=True)

print("Dataset loaded and indexed by date:")
print(df.head())
print("\nDataFrame Info:")
df.info()
print("\nMissing values in the dataset:")
print(df.isnull().sum())

1.2 Visualize original time series
plt.figure(figsize=(12, 6))
plt.plot(df['passengers'])
plt.title('AirPassengers Time Series')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.grid(True)
plt.show()

1.3 Apply Logarithmic Transformation
df['passengers_log'] = np.log(df['passengers'])

plt.figure(figsize=(12, 6))
plt.plot(df['passengers_log'])
plt.title('AirPassengers Time Series (Log Transformed)')
plt.xlabel('Date')
plt.ylabel('Log Number of Passengers')
plt.grid(True)
plt.show()
print("\nFirst 5 rows after log transformation:")
print(df.head())

1.4 Apply First-Order Differencing
df['passengers_diff'] = df['passengers_log'].diff()

plt.figure(figsize=(12, 6))
plt.plot(df['passengers_diff'])
plt.title('AirPassengers Time Series (Log Transformed and Differenced)')
plt.xlabel('Date')
plt.ylabel('Differenced Log Number of Passengers')
plt.grid(True)
plt.show()
print("\nFirst 5 rows after first-order differencing:")
print(df.head())

ADF test after first-order differencing
diff_series = df['passengers_diff'].dropna()
result_adf1 = adfuller(diff_series)
print(f"\nADF Statistic (First Differenced): {result_adf1[0]:.4f}")
print(f"p-value (First Differenced): {result_adf1[1]:.4f}")
if result_adf1[1] <= 0.05: print("Conclusion: Time series is likely stationary (after first differencing).")
else: print("Conclusion: Time series is likely non-stationary (after first differencing).")

1.5 Apply Seasonal Differencing (Lag 12)
df['passengers_diff_seasonal'] = df['passengers_diff'].diff(12)

plt.figure(figsize=(12, 6))
plt.plot(df['passengers_diff_seasonal'])
plt.title('AirPassengers Time Series (Log Transformed, Differenced, and Seasonally Differenced)')
plt.xlabel('Date')
plt.ylabel('Seasonally Differenced Log Number of Passengers')
plt.grid(True)
plt.show()
print("\nFirst 15 rows after seasonal differencing:")
print(df.head(15))

ADF test after seasonal differencing
diff_seasonal_series = df['passengers_diff_seasonal'].dropna()
result_adf2 = adfuller(diff_seasonal_series)
print(f"\nADF Statistic (Seasonally Differenced): {result_adf2[0]:.4f}")
print(f"p-value (Seasonally Differenced): {result_adf2[1]:.4f}")
if result_adf2[1] <= 0.05: print("Conclusion: Time series is likely stationary (after seasonal differencing).")
else: print("Conclusion: Time series is likely non-stationary (after seasonal differencing).")

1.6 Scale the data (MinMaxScaler)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = df['passengers_diff_seasonal'].dropna().values.reshape(-1, 1)
df_scaled = scaler.fit_transform(scaled_data)

df_scaled_series = pd.Series(df_scaled.flatten(), index=df['passengers_diff_seasonal'].dropna().index, name='passengers_scaled')
df = df.join(df_scaled_series)

print("\nMin and Max of scaled data:")
print(f"Min: {df_scaled.min():.4f}, Max: {df_scaled.max():.4f}")
print("\nFirst 5 rows after scaling:")
print(df[['passengers_diff_seasonal', 'passengers_scaled']].head(15))

1.7 Feature Engineering
df_features = df[['passengers_scaled']].copy()
for i in range(1, 13):
    df_features[f'passengers_scaled_lag_{i}'] = df_features['passengers_scaled'].shift(i)
df_features['year'] = df_features.index.year
df_features['month'] = df_features.index.month
df_features.dropna(inplace=True)

print("\nFirst 5 rows of feature-engineered dataset:")
print(df_features.head())
print(f"Shape of feature-engineered dataset: {df_features.shape}")

1.8 Data Splitting (Training, Validation, Test)
total_samples = len(df_features)
train_size = int(total_samples * 0.7)
val_size = int(total_samples * 0.15)
test_size = total_samples - train_size - val_size

train_data = df_features.iloc[:train_size]
val_data = df_features.iloc[train_size:train_size + val_size]
test_data = df_features.iloc[train_size + val_size:]

print(f"\nData split: Total samples={total_samples}, Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

==================================================================================
SECTION 2: SARIMA Benchmark Model Implementation
==================================================================================

Define SARIMA model orders
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

Extract relevant data for SARIMA
passengers_log_series = df['passengers_log']
sarima_train_log_val_set = passengers_log_series.loc[:train_data.index[-1]]
sarima_val_log_set = passengers_log_series.loc[val_data.index[0]:val_data.index[-1]]
actual_val_passengers = df['passengers'].loc[val_data.index[0]:val_data.index[-1]]
sarima_train_log_test_set = passengers_log_series.loc[:val_data.index[-1]]
sarima_test_log_set = passengers_log_series.loc[test_data.index[0]:test_data.index[-1]]
actual_test_passengers = df['passengers'].loc[test_data.index[0]:test_data.index[-1]]

print("\nSARIMA Data Preparation Complete.")

2.1 Train and Forecast for Validation Set
sarima_model_val = sm.tsa.statespace.SARIMAX(sarima_train_log_val_set, order=order, seasonal_order=seasonal_order)
sarima_model_val_fit = sarima_model_val.fit(disp=False)

sarima_val_preds_log = sarima_model_val_fit.get_forecast(steps=len(sarima_val_log_set)).predicted_mean
sarima_val_preds_log.index = sarima_val_log_set.index
sarima_val_preds_original_scale = np.exp(sarima_val_preds_log)

print("SARIMA validation forecast complete.")

2.2 Train and Forecast for Test Set
sarima_model_test = sm.tsa.statespace.SARIMAX(sarima_train_log_test_set, order=order, seasonal_order=seasonal_order)
sarima_model_test_fit = sarima_model_test.fit(disp=False)

sarima_test_preds_log = sarima_model_test_fit.get_forecast(steps=len(sarima_test_log_set)).predicted_mean
sarima_test_preds_log.index = sarima_test_log_set.index
sarima_test_preds_original_scale = np.exp(sarima_test_preds_log)

print("SARIMA test forecast complete.")

==================================================================================
SECTION 3: Attention-based Encoder-Decoder Model Development
==================================================================================

n_features = 3  # (passengers_scaled, year, month) per timestep
n_steps_in = 12 # Look-back window (encoder input sequence length)
n_steps_out = 12 # Forecasting horizon (decoder output sequence length)
latent_dim = 128 # Dimensionality of the LSTM's hidden state and cell state

3.1 Encoder
encoder_inputs = Input(shape=(n_steps_in, n_features), name='encoder_input')
encoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

3.2 Decoder
decoder_inputs = RepeatVector(n_steps_out)(state_h)
decoder_lstm = LSTM(latent_dim, return_sequences=True, name='decoder_lstm')
decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)

3.3 Attention Mechanism
attention_layer = tf.keras.layers.Attention(name='attention_mechanism')
attention_output = attention_layer([decoder_outputs, encoder_outputs])

decoder_combined_context = concatenate([decoder_outputs, attention_output], name='decoder_combined_context')
decoder_dense = TimeDistributed(Dense(1, activation='linear'), name='output_layer')
decoder_predictions = decoder_dense(decoder_combined_context)

3.4 Full Model
model = Model(inputs=encoder_inputs, outputs=decoder_predictions)
model.compile(optimizer='adam', loss='mse')

print("\nEncoder-Decoder model with attention defined and compiled.")
model.summary()

==================================================================================
SECTION 4: Prepare Data for Attention Model and Training
==================================================================================

4.1 Create data for NN sequences
data_for_nn = df[['passengers_scaled']].copy()
data_for_nn['year'] = data_for_nn.index.year
data_for_nn['month'] = data_for_nn.index.month
data_for_nn.dropna(inplace=True)
data_for_nn_sorted = data_for_nn.sort_index()

X_enc_all = []
y_dec_all = []

for i in range(n_steps_in, len(data_for_nn_sorted) - n_steps_out + 1):
    encoder_input_sequence = data_for_nn_sorted.iloc[i - n_steps_in:i].values
    decoder_target_sequence = data_for_nn_sorted['passengers_scaled'].iloc[i : i + n_steps_out].values
    X_enc_all.append(encoder_input_sequence)
    y_dec_all.append(decoder_target_sequence)

X_enc_all = np.array(X_enc_all)
y_dec_all = np.array(y_dec_all).reshape(y_dec_all.shape[0], y_dec_all.shape[1], 1)

print(f"\nShape of X_enc_all: {X_enc_all.shape}")
print(f"Shape of y_dec_all: {y_dec_all.shape}")

4.2 Split NN data
total_samples_nn = len(X_enc_all)
nn_train_size = int(total_samples_nn * 0.7)
nn_val_size = int(total_samples_nn * 0.15)
nn_test_size = total_samples_nn - nn_train_size - nn_val_size

X_train_enc = X_enc_all[:nn_train_size]
y_train_dec = y_dec_all[:nn_train_size]
X_val_enc = X_enc_all[nn_train_size : nn_train_size + nn_val_size]
y_val_dec = y_dec_all[nn_train_size : nn_train_size + nn_val_size]
X_test_enc = X_enc_all[nn_train_size + nn_val_size :]
y_test_dec = y_dec_all[nn_train_size + nn_val_size :]

print(f"NN Data split: Total={total_samples_nn}, Train={len(X_train_enc)}, Val={len(X_val_enc)}, Test={len(X_test_enc)}")

4.3 Train the Attention Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_attention_model.keras', monitor='val_loss', save_best_only=True)

history = model.fit(X_train_enc, y_train_dec,
                    epochs=50,
                    batch_size=16,
                    validation_data=(X_val_enc, y_val_dec),
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=0)

print("\nAttention model training complete.")

4.4 Generate predictions
y_val_pred_scaled = model.predict(X_val_enc, verbose=0)
y_test_pred_scaled = model.predict(X_test_enc, verbose=0)
print("Predictions generated for validation and test sets.")

4.5 Inverse Transformation Helper Function
def inverse_transform_single_value(scaled_value, sample_idx_in_subset, horizon_step, subset_type, df_original, df_features_index, nn_train_size, nn_val_size, scaler):
    if subset_type == 'val':
        base_idx_in_df_features = nn_train_size + sample_idx_in_subset
    elif subset_type == 'test':
        base_idx_in_df_features = nn_train_size + nn_val_size + sample_idx_in_subset
    else:
        raise ValueError("subset_type must be 'val' or 'test'")

    forecast_date = df_features_index[base_idx_in_df_features + horizon_step - 1]

    passengers_diff_seasonal_value = scaler.inverse_transform([[scaled_value]])[0][0]
    prev_diff_date = forecast_date - pd.DateOffset(months=12)
     Handle cases where prev_diff_date might be outside df_original, e.g., for the very first predictions.
     For this dataset, enough history is assumed to be present.
    prev_passengers_diff = df_original['passengers_diff'].loc[prev_diff_date]
    passengers_diff_value = passengers_diff_seasonal_value + prev_passengers_diff

    prev_log_date = forecast_date - pd.DateOffset(months=1)
    prev_passengers_log = df_original['passengers_log'].loc[prev_log_date]
    passengers_log_value = passengers_diff_value + prev_passengers_log

    passengers_original_value = np.exp(passengers_log_value)
    return passengers_original_value

4.6 Apply inverse transformation
nn_val_preds_h1_original = []
nn_val_actuals_h1_original = []
nn_val_preds_h12_original = []
nn_val_actuals_h12_original = []

for k in range(len(X_val_enc)):
    scaled_pred_h1 = y_val_pred_scaled[k, 0, 0]
    nn_val_preds_h1_original.append(inverse_transform_single_value(scaled_pred_h1, k, 1, 'val', df, df_features.index, nn_train_size, nn_val_size, scaler))
    scaled_actual_h1 = y_val_dec[k,0,0]
    nn_val_actuals_h1_original.append(inverse_transform_single_value(scaled_actual_h1, k, 1, 'val', df, df_features.index, nn_train_size, nn_val_size, scaler))

    if n_steps_out >= 12:
        scaled_pred_h12 = y_val_pred_scaled[k, 11, 0]
        nn_val_preds_h12_original.append(inverse_transform_single_value(scaled_pred_h12, k, 12, 'val', df, df_features.index, nn_train_size, nn_val_size, scaler))
        scaled_actual_h12 = y_val_dec[k, 11, 0]
        nn_val_actuals_h12_original.append(inverse_transform_single_value(scaled_actual_h12, k, 12, 'val', df, df_features.index, nn_train_size, nn_val_size, scaler))

nn_test_preds_h1_original = []
nn_test_actuals_h1_original = []
nn_test_preds_h12_original = []
nn_test_actuals_h12_original = []

for k in range(len(X_test_enc)):
    scaled_pred_h1 = y_test_pred_scaled[k, 0, 0]
    nn_test_preds_h1_original.append(inverse_transform_single_value(scaled_pred_h1, k, 1, 'test', df, df_features.index, nn_train_size, nn_val_size, scaler))
    scaled_actual_h1 = y_test_dec[k, 0, 0]
    nn_test_actuals_h1_original.append(inverse_transform_single_value(scaled_actual_h1, k, 1, 'test', df, df_features.index, nn_train_size, nn_val_size, scaler))

    if n_steps_out >= 12:
        scaled_pred_h12 = y_test_pred_scaled[k, 11, 0]
        nn_test_preds_h12_original.append(inverse_transform_single_value(scaled_pred_h12, k, 12, 'test', df, df_features.index, nn_train_size, nn_val_size, scaler))
        scaled_actual_h12 = y_test_dec[k, 11, 0]
        nn_test_actuals_h12_original.append(inverse_transform_single_value(scaled_actual_h12, k, 12, 'test', df, df_features.index, nn_train_size, nn_val_size, scaler))

nn_val_preds_h1_original = np.array(nn_val_preds_h1_original)
nn_val_actuals_h1_original = np.array(nn_val_actuals_h1_original)
nn_val_preds_h12_original = np.array(nn_val_preds_h12_original)
nn_val_actuals_h12_original = np.array(nn_val_actuals_h12_original)

nn_test_preds_h1_original = np.array(nn_test_preds_h1_original)
nn_test_actuals_h1_original = np.array(nn_test_actuals_h1_original)
nn_test_preds_h12_original = np.array(nn_test_preds_h12_original)
nn_test_actuals_h12_original = np.array(nn_test_actuals_h12_original)

print("Inverse transformation for NN predictions and actuals complete.")


4.7 Performance Metrics (Custom MAPE function)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_indices = y_true != 0
    if not np.any(nonzero_indices): return np.nan
    return np.mean(np.abs((y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices])) * 100

def calculate_metrics(y_true, y_pred, horizon_len):
    y_true_h = y_true.iloc[:horizon_len] if isinstance(y_true, pd.Series) else y_true[:horizon_len]
    y_pred_h = y_pred.iloc[:horizon_len] if isinstance(y_pred, pd.Series) else y_pred[:horizon_len]

    rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
    mae = mean_absolute_error(y_true_h, y_pred_h)
    mape = mean_absolute_percentage_error(y_true_h, y_pred_h)
    return rmse, mae, mape

Calculate SARIMA metrics
sarima_val_rmse_h1, sarima_val_mae_h1, sarima_val_mape_h1 = calculate_metrics(actual_val_passengers, sarima_val_preds_original_scale, 1)
sarima_val_rmse_h12, sarima_val_mae_h12, sarima_val_mape_h12 = calculate_metrics(actual_val_passengers, sarima_val_preds_original_scale, 12)
sarima_test_rmse_h1, sarima_test_mae_h1, sarima_test_mape_h1 = calculate_metrics(actual_test_passengers, sarima_test_preds_original_scale, 1)
sarima_test_rmse_h12, sarima_test_mae_h12, sarima_test_mape_h12 = calculate_metrics(actual_test_passengers, sarima_test_preds_original_scale, 12)

Calculate NN metrics
nn_val_rmse_h1, nn_val_mae_h1, nn_val_mape_h1 = calculate_metrics(nn_val_actuals_h1_original, nn_val_preds_h1_original, len(nn_val_actuals_h1_original))
nn_val_rmse_h12, nn_val_mae_h12, nn_val_mape_h12 = calculate_metrics(nn_val_actuals_h12_original, nn_val_preds_h12_original, len(nn_val_actuals_h12_original))
nn_test_rmse_h1, nn_test_mae_h1, nn_test_mape_h1 = calculate_metrics(nn_test_actuals_h1_original, nn_test_preds_h1_original, len(nn_test_actuals_h1_original))
nn_test_rmse_h12, nn_test_mae_h12, nn_test_mape_h12 = calculate_metrics(nn_test_actuals_h12_original, nn_test_preds_h12_original, len(nn_test_actuals_h12_original))

print("\nPerformance metrics calculated.")

4.8 Visualize NN Predictions
val_dates_for_nn_h1_preds = df_features.index[nn_train_size : nn_train_size + nn_val_size]
test_dates_for_nn_h1_preds = df_features.index[nn_train_size + nn_val_size : nn_train_size + nn_val_size + nn_test_size]

nn_val_preds_h1_series = pd.Series(nn_val_preds_h1_original, index=val_dates_for_nn_h1_preds)
nn_test_preds_h1_series = pd.Series(nn_test_preds_h1_original, index=test_dates_for_nn_h1_preds)

plt.figure(figsize=(14, 7))
plt.plot(actual_val_passengers.index, actual_val_passengers, label='Actual Passengers (Validation)', color='blue', marker='.')
plt.plot(sarima_val_preds_original_scale.index, sarima_val_preds_original_scale, label='SARIMA Predictions (Validation)', color='red', linestyle='--')
plt.plot(nn_val_preds_h1_series.index, nn_val_preds_h1_series, label='NN Attention Predictions (Validation, h=1)', color='green', linestyle=':')
plt.title('SARIMA & NN Attention Models: Actual vs Predicted Passengers (Validation Set)')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(actual_test_passengers.index, actual_test_passengers, label='Actual Passengers (Test)', color='blue', marker='.')
plt.plot(sarima_test_preds_original_scale.index, sarima_test_preds_original_scale, label='SARIMA Predictions (Test)', color='red', linestyle='--')
plt.plot(nn_test_preds_h1_series.index, nn_test_preds_h1_series, label='NN Attention Predictions (Test, h=1)', color='green', linestyle=':')
plt.title('SARIMA & NN Attention Models: Actual vs Predicted Passengers (Test Set)')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid(True)
plt.show()

==================================================================================
SECTION 5: Attention Weight Analysis
==================================================================================

Recreate attention_model for visualization
query_tensor = decoder_outputs
key_tensor = encoder_outputs
attention_scores_tensor = Dot(axes=(2, 2))([query_tensor, key_tensor])
attention_weights = Activation('softmax', name='attention_weights')(attention_scores_tensor)
attention_model = Model(inputs=encoder_inputs, outputs=attention_weights)

Select a few samples from X_test_enc for visualization
sample_indices = [0, 1, 2]  # Example indices
selected_X_test_enc = X_test_enc[sample_indices]
predicted_attention_weights = attention_model.predict(selected_X_test_enc, verbose=0)

print("\nVisualizing attention weights for selected test samples...")
plt.figure(figsize=(15, 5 * len(sample_indices)))

for i, sample_idx in enumerate(sample_indices):
    plt.subplot(len(sample_indices), 1, i + 1)
    sns.heatmap(predicted_attention_weights[i], cmap='viridis', cbar=True)
    plt.title(f'Attention Weights for Test Sample {sample_idx + 1}')
    plt.xlabel('Encoder Input Timesteps (t-12 to t-1)')
    plt.ylabel('Decoder Output Timesteps (t to t+11)')

plt.tight_layout()
plt.show()
print("Attention weights visualized.")


2. Text-Based Analytical Report

Data Analysis Key Findings

Data Preparation (AirPassengers Dataset):
  The AirPassengers dataset was loaded, preprocessed, and transformed from a float-based time column to a DatetimeIndex with 144 monthly entries spanning from January 1949 to December 1960.
       Initial visualization showed an upward trend and increasing variance, indicating clear non-stationarity.
       A natural logarithm transformation was applied to stabilize the variance, making the fluctuations more consistent over time.
       First-order differencing (lag 1) was applied to remove the trend. An Augmented Dickey-Fuller (ADF) test on the first-differenced series yielded a p-value of 0.0711, suggesting that the series was still non-stationary, likely due to a persistent seasonal component.
       Subsequent seasonal differencing (lag 12, appropriate for monthly data) successfully achieved stationarity, with the ADF test producing a p-value of 0.0002, well below the 0.05 significance level.
       The stationary series (passengers_diff_seasonal) was then scaled using MinMaxScaler to a range of (0, 1). The scaler object was preserved for inverse transformations.
       Feature engineering involved creating 12 lagged versions of the `passengers_scaled` variable to capture past dependencies (a common practice for sequence models), along with `year` and `month` features to help the models account for remaining seasonality and long-term patterns.
      The data was chronologically split into training (70%), validation (15%), and test (15%) sets, ensuring no future information leaked into past observations. This resulted in: `train_data` (83 samples), `val_data` (17 samples), and `test_data` (19 samples).

SARIMA Benchmark Model Performance:
       A Seasonal AutoRegressive Integrated Moving Average (SARIMA) model with `order=(1,1,1)` and `seasonal_order=(1,1,1,12)` was chosen as the benchmark, reflecting the observed non-seasonal (d=1) and seasonal (D=1) differencing. The (p, q) and (P, Q) orders were chosen as a reasonable starting point.
       The model was trained separately for validation and test periods, using the log-transformed series to align with the preprocessing steps.
       A `ConvergenceWarning` was occasionally observed during model fitting, indicating potential issues with optimization or model complexity, but predictions were still generated.

Attention-based Encoder-Decoder Model Development:
      An Encoder-Decoder sequence-to-sequence deep learning model was implemented using LSTM layers in TensorFlow/Keras.
    Hyperparameters: `n_steps_in = 12` (encoder look-back window), `n_steps_out = 12` (decoder forecast horizon), `n_features = 3` (representing `passengers_scaled`, `year`, `month` per timestep), and `latent_dim = 128` (LSTM state dimensionality).
    Architecture:
        Encoder: An LSTM layer taking input sequences of shape `(None, n_steps_in, n_features)` and returning its last hidden state (`state_h`), cell state (`state_c`), and outputs for each timestep (`encoder_outputs`).
        Decoder: The encoder's final hidden state (`state_h`) was repeated `n_steps_out` times to serve as input to the decoder. A separate LSTM layer was initialized with the encoder's final states (`state_h`, `state_c`) and produced `decoder_outputs` for each forecast step.
        Self-Attention Mechanism: A Keras `Attention` layer was incorporated, computing attention scores between `decoder_outputs` (query) and `encoder_outputs` (value/key). The resulting context vector was concatenated with `decoder_outputs`.
        Output Layer: A `TimeDistributed(Dense(1, activation='linear'))` layer processed the combined context to generate the final `n_steps_out` predictions.
    Model Summary: The combined model had a total of 199,425 trainable parameters.
    Hyperparameter Tuning Strategy: For this project, a direct hyperparameter tuning loop using rolling cross-validation was not explicitly implemented due to time constraints but a fixed set of hyperparameters for the attention model (latent_dim, epochs, batch_size) were chosen as a reasonable starting point. Early stopping based on validation loss was used to prevent overfitting and implicitly manage the number of training epochs.

Attention-based Encoder-Decoder Model Performance:
       Data for the neural network was structured into input sequences (`X_enc_all`) of shape `(108, 12, 3)` and target sequences (`y_dec_all`) of shape `(108, 12, 1)`. These were split into NN training (75 samples), validation (16 samples), and test (17 samples) sets.
       The model was compiled with the Adam optimizer and Mean Squared Error (`mse`) loss function.
       Training utilized `EarlyStopping` with `patience=10` and `ModelCheckpoint` to save the best weights based on validation loss.
       A custom `inverse_transform_single_value` function was developed to accurately revert the scaled, differenced, and log-transformed predictions and actuals back to the original passenger count scale. This function meticulously reverses each preprocessing step.

### Comparative Performance Analysis

Performance Metrics (RMSE, MAE, MAPE)

| Model | Set | Horizon | RMSE (Lower is Better) | MAE (Lower is Better) | MAPE (Lower is Better) | 
| :---- | :--- | :------ | :-------------------- | :-------------------- | :-------------------- | 
| SARIMA | Validation | h=1 | 11.21 | 11.21 | 3.30% |
| SARIMA | Validation | h=12 | 32.56 | 30.19 | 8.27% |
| **NN Attention** | **Validation** | **h=1** | **7.75** | **6.42** | **1.61%** |
| **NN Attention** | **Validation** | **h=12** | **15.22** | **12.67** | **3.06%** |
| SARIMA | Test | h=1 | 20.04 | 20.04 | 4.25% |
| SARIMA | Test | h=12 | 14.05 | 9.97 | 2.26% |
| NN Attention | Test | h=1 | 15.43 | 12.94 | 3.15% |
| NN Attention | Test | h=12 | 20.69 | 15.26 | 3.39% |


### 3. Textual Description and Interpretation of Learned Attention Mechanism Weights

The heatmaps visualize the attention weights for three selected test samples. The x-axis represents the encoder input timesteps, from `t-12` (past 12 months, index 0) to `t-1` (past 1 month, index 11). The y-axis represents the decoder output timesteps, from `t` (current month, index 0) to `t+11` (future 11 months, index 11).

Observations:

1.  Diagonal Pattern (Recency Bias): For predictions closer to the current time (`t`, `t+1`, `t+2`), the model tends to place higher attention on the most recent encoder input timesteps (`t-1`, `t-2`). This is visible as a diagonal line of slightly higher intensity in the top-right corner of the heatmaps, although it's not sharply defined.

2.  Weak but Persistent Seasonal Pattern: While not overwhelmingly strong, there appears to be a subtle tendency for the model to attend to timesteps 12 months prior (index 0 on the x-axis) for certain predictions. This suggests a recognition of annual seasonality, but it's not the dominant pattern.

3.  Diffuse Attention: Overall, the attention weights are quite diffused across many past timesteps. This indicates that the model considers a broad range of past information rather than focusing sharply on one or two specific points. The color intensity is relatively uniform across much of the heatmap, implying that no single past month strongly dictates any single future prediction.

4.  Lack of Strong Localized Attention: Unlike models that might heavily focus on the exact same month in the previous year for a given prediction (e.g., `t` attends strongly to `t-12`), this model's attention is distributed. This might be a consequence of the strong differencing and scaling applied, which removes much of the direct linear and seasonal dependencies, forcing the model to find more complex, non-linear relationships across the entire input sequence.

Interpretation:

The attention mechanism in this model seems to implement a form of 'soft' or 'diffuse' attention. Instead of identifying a few critical past timesteps, it appears to aggregate information from the entire look-back window. This could be beneficial in complex time series where dependencies are not always clear-cut or single-point driven. The slightly elevated attention on recent past values is intuitive, as the immediate past usually holds significant predictive power. The faint seasonal hints suggest the attention mechanism might be weakly capturing the remaining seasonal information after differencing, or it's combining with the time-based features (`year`, `month`) to infer seasonality indirectly.

It's important to note that the model is making predictions on the *scaled, seasonally differenced, and first-order differenced log-transformed data. This preprocessing significantly alters the nature of the time series, and therefore, the attention patterns might reflect the more subtle relationships within this transformed space rather than direct raw data patterns.

 4. Summary of Project Findings, Comparison, and Discussion

 Project Findings:

Data Preparation:The `AirPassengers` dataset required significant preprocessing including log transformation, first-order differencing, and seasonal differencing to achieve stationarity, which is crucial for traditional time series models like SARIMA and beneficial for deep learning models.
*   Benchmark Model: The SARIMA model provided a solid baseline, capturing both trend and seasonality in the log-transformed data.
*   Attention-based Model: An Encoder-Decoder LSTM with self-attention was successfully built and trained to forecast the transformed passenger data.
*   Performance: Both models were evaluated using RMSE, MAE, and MAPE for short (h=1) and medium (h=12) horizons on validation and test sets.

Comparison of Model Performances:

Looking at the performance tables, the Attention-based Encoder-Decoder model generally outperformed the SARIMA benchmark model on the validation set, especially for the short-term forecast (h=1) (NN MAPE: 1.61% vs SARIMA MAPE: 3.30%). For the validation set, the NN model consistently showed lower RMSE, MAE, and MAPE across both horizons. This indicates the potential of deep learning models to capture more complex patterns when enough data is available and tuned properly.

However, on the test set, the SARIMA model demonstrated superior performance for the h=12 horizon (SARIMA MAPE: 2.26% vs NN MAPE: 3.39%), while the NN Attention model performed slightly better for h=1 (NN MAPE: 3.15% vs SARIMA MAPE: 4.25%). The SARIMA model's relative strength on the test set for longer horizons might suggest a more robust generalization to unseen data under certain conditions, or that the deep learning model might be more sensitive to the limited size of the dataset for training after sequence creation, or require more extensive hyperparameter tuning.

 Discussion of Insights:

Preprocessing Impact: The extensive preprocessing (log transformation, double differencing, scaling) was critical for stabilizing the time series, making it amenable to both SARIMA and neural network models. However, this also means the models are learning patterns in a highly transformed space, which complicates direct interpretability.
Attention Mechanism Insights: The attention weight analysis revealed a **diffuse attention pattern**, indicating that the model considers a broad range of past timesteps rather than focusing intensely on a few. A subtle **recency bias** was observed for immediate forecasts, and a **weak seasonal pattern** suggested some recognition of annual dependencies even after explicit seasonal differencing. This diffuse nature is likely a direct consequence of the aggressive differencing, which removed strong, obvious dependencies. The attention mechanism here acts more as an aggregator of diverse past information rather than a highlighter of singular critical events.
  Model Complexity vs. Data Size: The attention-based deep learning model, while powerful, requires more data and careful tuning. Its performance variability between validation and test sets (especially for h=12) suggests it might be prone to overfitting or could benefit from more data, more robust time series cross-validation strategies (like rolling origin), or further hyperparameter optimization.
  Future Work: Next steps would include implementing a formal rolling origin cross-validation strategy for hyperparameter tuning of the attention model, exploring different deep learning architectures (e.g., Transformers), and experimenting with alternative preprocessing techniques to see how they influence both predictive performance and attention patterns. It would also be insightful to compare attention mechanisms directly on raw or less transformed data to observe if more localized attention patterns emerge.
