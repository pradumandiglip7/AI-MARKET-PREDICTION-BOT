# # import os
# # import joblib
# # import numpy as np
# # import pandas as pd
# # import tensorflow as tf
# # from sklearn.preprocessing import MinMaxScaler
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# # # --- CONFIGURATION (Match this to your data) ---
# # # If using HOURLY data, LOOKBACK=60 means "Past 60 Hours" (2.5 days context)
# # # If using DAILY data, LOOKBACK=60 means "Past 60 Days" (2 months context)
# # LOOKBACK_WINDOW = 60 

# # # Columns the AI will "Read" to make a decision
# # FEATURE_COLUMNS = ['close', 'rsi', 'macd', 'macd_signal', 'williams_r', 'momentum', 'ema_20']

# # def load_data(csv_path):
# #     print(f"⏳ Loading dataset from: {csv_path}")
# #     if not os.path.exists(csv_path):
# #         raise FileNotFoundError(f"❌ File not found: {csv_path}")
        
# #     df = pd.read_csv(csv_path)
    
# #     # Drop rows with NaN (empty) values to prevent crashes
# #     df.dropna(inplace=True)
    
# #     # 1. Prepare Inputs (X) and Output (y)
# #     # We filter only the columns we need
# #     data_x = df[FEATURE_COLUMNS].values
# #     data_y = df['target'].values
    
# #     # 2. Scale the Data (Crucial for LSTM)
# #     # Neural Networks fail if numbers are too big (like 50000). We squeeze them between 0 and 1.
# #     scaler = MinMaxScaler(feature_range=(0, 1))
# #     data_x_scaled = scaler.fit_transform(data_x)
    
# #     # Save the scaler! The Live Bot needs this to understand live prices.
# #     os.makedirs("models", exist_ok=True)
# #     joblib.dump(scaler, "models/scaler.pkl")
# #     print("✅ Scaler saved to models/scaler.pkl")
    
# #     return data_x_scaled, data_y

# # def create_sequences(data, target, lookback):
# #     """
# #     Converts linear data into 3D sequences for LSTM.
# #     Input Shape: (Samples, Time Steps, Features)
# #     """
# #     X, y = [], []
# #     for i in range(lookback, len(data)):
# #         # Take a window of 'lookback' steps (e.g., past 60 hours)
# #         X.append(data[i-lookback:i])
# #         # Take the target of the CURRENT step
# #         y.append(target[i])
        
# #     return np.array(X), np.array(y)

# # def build_model(input_shape):
# #     """
# #     Builds a Professional LSTM Architecture
# #     """
# #     model = Sequential()
    
# #     # Layer 1: LSTM with Return Sequences (Passes memory to next layer)
# #     model.add(Input(shape=input_shape))
# #     model.add(LSTM(units=64, return_sequences=True))
# #     model.add(Dropout(0.2)) # Prevents Overfitting (Memorizing instead of learning)
    
# #     # Layer 2: LSTM (Final memory layer)
# #     model.add(LSTM(units=64, return_sequences=False))
# #     model.add(Dropout(0.2))
    
# #     # Layer 3: Dense (Decision Making)
# #     model.add(Dense(units=32, activation='relu'))
    
# #     # Output Layer: Sigmoid (Returns probability between 0 and 1)
# #     # < 0.5 = DOWN, > 0.5 = UP
# #     model.add(Dense(units=1, activation='sigmoid'))
    
# #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# #     return model

# # def main():
# #     # Path to your Training Data
# #     # NOTE: Ideally, use the 'Hourly' file for a Real-Time Bot.
# #     csv_file = "data/training/full_history_data.csv"
    
# #     # Fail-safe check
# #     if not os.path.exists(csv_file):
# #         # Check if file is in current directory
# #         if os.path.exists("full_history_data.csv"):
# #             csv_file = "full_history_data.csv"
    
# #     try:
# #         # 1. Process Data
# #         X_data, y_data = load_data(csv_file)
        
# #         print("✂️ Creating Time Sequences (This may take a moment)...")
# #         X, y = create_sequences(X_data, y_data, LOOKBACK_WINDOW)
        
# #         # 2. Split Train (80%) / Test (20%)
# #         # We don't use random split because Time Series must remain in order!
# #         split_idx = int(len(X) * 0.8)
# #         X_train, X_test = X[:split_idx], X[split_idx:]
# #         y_train, y_test = y[:split_idx], y[split_idx:]
        
# #         print(f"📊 Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")
        
# #         # 3. Build & Train
# #         model = build_model((X_train.shape[1], X_train.shape[2]))
        
# #         print("\n🧠 Training the Brain (Epochs=5)...")
# #         model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
        
# #         # 4. Save
# #         model_path = "models/prediction_model.keras"
# #         model.save(model_path)
# #         print(f"\n🎉 SUCCESS! Model saved to: {model_path}")
# #         print("Next Step: Phase 5 - Connect this Brain to the Live Bot.")
        
# #     except Exception as e:
# #         print(f"\n❌ Critical Error: {e}")
# #         print("Tip: Did you generate the CSV file in the previous step?")

# # if __name__ == "__main__":
# #     main()





# import os
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# # --- CONFIGURATION ---
# LOOKBACK_WINDOW = 60 
# FEATURE_COLUMNS = ['close', 'rsi', 'macd', 'macd_signal', 'williams_r', 'momentum', 'ema_20']

# def load_data(csv_path, scaler_save_path):
#     print(f"⏳ Loading dataset from: {csv_path}")
#     if not os.path.exists(csv_path):
#         print(f"❌ File not found: {csv_path}")
#         return None, None
        
#     df = pd.read_csv(csv_path)
#     df.dropna(inplace=True)
    
#     # Select Features & Target
#     data_x = df[FEATURE_COLUMNS].values
#     data_y = df['target'].values
    
#     # Scale Data (0 to 1)
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_x_scaled = scaler.fit_transform(data_x)
    
#     # Save the specific scaler for this timeframe
#     os.makedirs("models", exist_ok=True)
#     joblib.dump(scaler, scaler_save_path)
#     print(f"✅ Scaler saved to {scaler_save_path}")
    
#     return data_x_scaled, data_y

# def create_sequences(data, target, lookback):
#     X, y = [], []
#     for i in range(lookback, len(data)):
#         X.append(data[i-lookback:i])
#         y.append(target[i])
#     return np.array(X), np.array(y)

# def build_model(input_shape):
#     model = Sequential()
#     # Thoda chhota model 5m/15m ke liye fast training ke liye
#     model.add(Input(shape=input_shape))
#     model.add(LSTM(units=50, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=50, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=25, activation='relu'))
#     model.add(Dense(units=1, activation='sigmoid'))
    
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def train_specific_timeframe(csv_file, model_name, scaler_name):
#     print(f"\n==========================================")
#     print(f"🚀 STARTING TRAINING: {model_name}")
#     print(f"==========================================")
    
#     # 1. Load Data
#     X_data, y_data = load_data(csv_file, scaler_name)
#     if X_data is None: return
    
#     # 2. Sequence
#     print("✂️ Creating Sequences...")
#     X, y = create_sequences(X_data, y_data, LOOKBACK_WINDOW)
    
#     # 3. Split
#     split = int(len(X) * 0.8)
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]
    
#     # 4. Train
#     model = build_model((X_train.shape[1], X_train.shape[2]))
#     print(f"🧠 Training {model_name}...")
#     model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    
#     # 5. Save
#     model.save(model_name)
#     print(f"🎉 SAVED: {model_name}")

# def main():
#     # --- TRAIN 5 MINUTE MODEL ---
#     train_specific_timeframe(
#         csv_file="data/training/data_5m.csv", 
#         model_name="models/model_5m.keras", 
#         scaler_name="models/scaler_5m.pkl"
#     )

#     # --- TRAIN 15 MINUTE MODEL ---
#     train_specific_timeframe(
#         csv_file="data/training/data_15m.csv", 
#         model_name="models/model_15m.keras", 
#         scaler_name="models/scaler_15m.pkl"
#     )
    
#     print("\n✅ ALL MODELS TRAINED SUCCESSFULLY!")

# if __name__ == "__main__":
#     main()









# Perplexity Improvement Version
# import os
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.utils.class_weight import compute_class_weight
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Bidirectional
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2


# # ============================================================
# # CONFIGURATION
# # ============================================================
# LOOKBACK_WINDOW = 60

# # Features the model will use to learn patterns
# FEATURE_COLUMNS = [
#     'close', 'rsi', 'macd', 'macd_signal', 'williams_r', 'momentum', 'ema_20'
# ]

# # Hyperparameters
# LSTM_UNITS_1 = 128
# LSTM_UNITS_2 = 64
# DENSE_UNITS = 32
# DROPOUT_RATE = 0.3
# RECURRENT_DROPOUT = 0.2
# LEARNING_RATE = 0.001
# BATCH_SIZE = 64
# EPOCHS = 50
# L2_REG = 0.001

# # ============================================================
# # TARGET LABELING STRATEGIES
# # ============================================================

# def create_target_threshold(df, threshold_pct=0.15, lookahead=5):
#     """
#     Strategy 1: Threshold-based labeling
#     - Predicts if price will move UP by at least `threshold_pct`% in next `lookahead` candles
#     - Ignores small/noisy moves (neither strong up nor down)
    
#     Parameters:
#     - threshold_pct: Minimum percentage move to consider (e.g., 0.15 = 0.15%)
#     - lookahead: Number of candles to look ahead
#     """
#     print(f"📊 Creating target: Threshold={threshold_pct}%, Lookahead={lookahead} candles")
    
#     # Calculate future return over `lookahead` candles
#     df['future_close'] = df['close'].shift(-lookahead)
#     df['future_return'] = (df['future_close'] - df['close']) / df['close'] * 100  # in percentage
    
#     # Label: 1 if return > threshold (BUY signal), 0 otherwise
#     df['target'] = (df['future_return'] > threshold_pct).astype(int)
    
#     # Drop rows where we can't calculate future return
#     df.dropna(subset=['target', 'future_return'], inplace=True)
    
#     # Clean up temporary columns
#     df.drop(columns=['future_close', 'future_return'], inplace=True)
    
#     return df


# def create_target_triple_barrier(df, profit_target=0.3, stop_loss=0.2, max_holding=10):
#     """
#     Strategy 2: Triple Barrier Method (Professional Quant Approach)
#     - Sets profit target (take profit), stop loss, and max holding period
#     - Labels based on which barrier is hit first
    
#     Parameters:
#     - profit_target: Take profit percentage (e.g., 0.3 = 0.3%)
#     - stop_loss: Stop loss percentage (e.g., 0.2 = 0.2%)
#     - max_holding: Maximum candles to hold before timeout
#     """
#     print(f"📊 Creating target: Triple Barrier (TP={profit_target}%, SL={stop_loss}%, MaxHold={max_holding})")
    
#     targets = []
#     closes = df['close'].values
    
#     for i in range(len(df)):
#         if i + max_holding >= len(df):
#             targets.append(np.nan)
#             continue
            
#         entry_price = closes[i]
#         tp_price = entry_price * (1 + profit_target / 100)
#         sl_price = entry_price * (1 - stop_loss / 100)
        
#         label = 0  # Default: no clear signal (or stop loss hit)
        
#         for j in range(1, max_holding + 1):
#             future_price = closes[i + j]
            
#             # Check if take profit hit first
#             if future_price >= tp_price:
#                 label = 1  # WIN - BUY signal was correct
#                 break
#             # Check if stop loss hit
#             elif future_price <= sl_price:
#                 label = 0  # LOSS - should not have bought
#                 break
        
#         targets.append(label)
    
#     df['target'] = targets
#     df.dropna(subset=['target'], inplace=True)
#     df['target'] = df['target'].astype(int)
    
#     return df


# def create_target_trend_following(df, ema_short=5, ema_long=20):
#     """
#     Strategy 3: Trend Following
#     - Labels based on whether short EMA crosses above long EMA (uptrend starting)
#     - More suitable for trend-following strategies
    
#     Parameters:
#     - ema_short: Short EMA period
#     - ema_long: Long EMA period
#     """
#     print(f"📊 Creating target: Trend Following (EMA{ema_short} vs EMA{ema_long})")
    
#     df['ema_short'] = df['close'].ewm(span=ema_short, adjust=False).mean()
#     df['ema_long'] = df['close'].ewm(span=ema_long, adjust=False).mean()
    
#     # Shift to make it predictive (we want to predict FUTURE crossover)
#     df['future_ema_short'] = df['ema_short'].shift(-5)
#     df['future_ema_long'] = df['ema_long'].shift(-5)
    
#     # Label: 1 if short EMA will be above long EMA (uptrend)
#     df['target'] = (df['future_ema_short'] > df['future_ema_long']).astype(int)
    
#     # Clean up
#     df.drop(columns=['ema_short', 'ema_long', 'future_ema_short', 'future_ema_long'], inplace=True)
#     df.dropna(subset=['target'], inplace=True)
    
#     return df


# # ============================================================
# # FEATURE ENGINEERING
# # ============================================================

# def add_extra_features(df):
#     """
#     Add more meaningful features that may help the model learn patterns.
#     """
#     print("🔧 Adding extra technical features...")
    
#     # Price changes
#     df['price_change_1'] = df['close'].pct_change(1) * 100
#     df['price_change_5'] = df['close'].pct_change(5) * 100
#     df['price_change_10'] = df['close'].pct_change(10) * 100
    
#     # Volatility (rolling std)
#     df['volatility_10'] = df['close'].rolling(window=10).std()
#     df['volatility_20'] = df['close'].rolling(window=20).std()
    
#     # RSI momentum (RSI change)
#     if 'rsi' in df.columns:
#         df['rsi_change'] = df['rsi'].diff(3)
    
#     # MACD histogram
#     if 'macd' in df.columns and 'macd_signal' in df.columns:
#         df['macd_histogram'] = df['macd'] - df['macd_signal']
    
#     # Price relative to EMA
#     if 'ema_20' in df.columns:
#         df['price_vs_ema'] = (df['close'] - df['ema_20']) / df['ema_20'] * 100
    
#     # High-Low range (if available)
#     if 'high' in df.columns and 'low' in df.columns:
#         df['hl_range'] = (df['high'] - df['low']) / df['close'] * 100
    
#     # Drop NaN rows created by rolling calculations
#     df.dropna(inplace=True)
    
#     return df


# # ============================================================
# # DATA LOADING AND PREPROCESSING
# # ============================================================

# def load_and_prepare_data(csv_path, scaler_save_path, target_strategy='threshold'):
#     """
#     Load data, create target, add features, and scale.
    
#     Parameters:
#     - csv_path: Path to CSV file
#     - scaler_save_path: Where to save the scaler
#     - target_strategy: 'threshold', 'triple_barrier', or 'trend'
#     """
#     print(f"\n⏳ Loading dataset from: {csv_path}")
    
#     if not os.path.exists(csv_path):
#         print(f"❌ File not found: {csv_path}")
#         return None, None

#     df = pd.read_csv(csv_path)
#     print(f"📊 Loaded {len(df)} rows")
    
#     # ---- STEP 1: Add Extra Features ----
#     df = add_extra_features(df)
    
#     # ---- STEP 2: Create Target Variable ----
#     if target_strategy == 'threshold':
#         # Good for 5m/15m data - predicts 0.15%+ move in next 5 candles
#         df = create_target_threshold(df, threshold_pct=0.15, lookahead=5)
#     elif target_strategy == 'triple_barrier':
#         # Professional approach - simulates actual trading
#         df = create_target_triple_barrier(df, profit_target=0.3, stop_loss=0.2, max_holding=10)
#     elif target_strategy == 'trend':
#         # Trend following - predicts trend direction
#         df = create_target_trend_following(df, ema_short=5, ema_long=20)
#     else:
#         print(f"❌ Unknown target strategy: {target_strategy}")
#         return None, None
    
#     # ---- STEP 3: Define Feature Columns ----
#     # Base features + new features we added
#     feature_cols = FEATURE_COLUMNS.copy()
    
#     # Add extra features if they exist
#     extra_features = [
#         'price_change_1', 'price_change_5', 'price_change_10',
#         'volatility_10', 'volatility_20', 'rsi_change',
#         'macd_histogram', 'price_vs_ema', 'hl_range'
#     ]
#     for feat in extra_features:
#         if feat in df.columns:
#             feature_cols.append(feat)
    
#     print(f"📊 Using {len(feature_cols)} features: {feature_cols}")
    
#     # Check all required columns exist
#     missing_cols = [col for col in feature_cols if col not in df.columns]
#     if missing_cols:
#         print(f"⚠️ Missing columns (will skip): {missing_cols}")
#         feature_cols = [col for col in feature_cols if col in df.columns]
    
#     # ---- STEP 4: Final Cleanup ----
#     df.dropna(inplace=True)
#     print(f"📊 After cleanup: {len(df)} rows")
    
#     if len(df) < LOOKBACK_WINDOW + 100:
#         print(f"❌ Not enough data. Need at least {LOOKBACK_WINDOW + 100} rows.")
#         return None, None
    
#     # ---- STEP 5: Check Class Balance ----
#     class_counts = df['target'].value_counts()
#     print(f"📊 Class distribution:\n{class_counts}")
#     print(f"📊 Class ratio: {class_counts.min() / class_counts.max():.2%}")
    
#     # Warn if severely imbalanced
#     if class_counts.min() / class_counts.max() < 0.3:
#         print("⚠️ Warning: Classes are imbalanced. Consider adjusting threshold.")
    
#     # ---- STEP 6: Scale Features ----
#     data_x = df[feature_cols].values
#     data_y = df['target'].values
    
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_x_scaled = scaler.fit_transform(data_x)
    
#     # Save scaler and feature columns
#     os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
#     joblib.dump({
#         'scaler': scaler,
#         'feature_columns': feature_cols
#     }, scaler_save_path)
#     print(f"✅ Scaler saved to {scaler_save_path}")
    
#     return data_x_scaled, data_y


# # ============================================================
# # SEQUENCE CREATION
# # ============================================================

# def create_sequences(data, target, lookback):
#     """Create time sequences for LSTM."""
#     X, y = [], []
#     total = len(data) - lookback
    
#     for i in range(lookback, len(data)):
#         X.append(data[i - lookback:i])
#         y.append(target[i])
        
#         if (i - lookback) % 10000 == 0 and i > lookback:
#             print(f"   Processed {i - lookback}/{total} sequences...")

#     return np.array(X), np.array(y)


# # ============================================================
# # MODEL BUILDING
# # ============================================================

# def build_optimized_model(input_shape):
#     """Build optimized LSTM model."""
#     model = Sequential([
#         Input(shape=input_shape),
        
#         Bidirectional(LSTM(
#             units=LSTM_UNITS_1,
#             return_sequences=True,
#             kernel_regularizer=l2(L2_REG),
#             recurrent_dropout=RECURRENT_DROPOUT
#         )),
#         BatchNormalization(),
#         Dropout(DROPOUT_RATE),

#         LSTM(
#             units=LSTM_UNITS_2,
#             return_sequences=False,
#             kernel_regularizer=l2(L2_REG),
#             recurrent_dropout=RECURRENT_DROPOUT
#         ),
#         BatchNormalization(),
#         Dropout(DROPOUT_RATE),

#         Dense(units=DENSE_UNITS, activation='relu', kernel_regularizer=l2(L2_REG)),
#         BatchNormalization(),
#         Dropout(DROPOUT_RATE / 2),

#         Dense(units=1, activation='sigmoid')
#     ])

#     optimizer = Adam(learning_rate=LEARNING_RATE)
    
#     model.compile(
#         optimizer=optimizer,
#         loss='binary_crossentropy',
#         metrics=[
#             'accuracy',
#             tf.keras.metrics.AUC(name='auc'),
#             tf.keras.metrics.Precision(name='precision'),
#             tf.keras.metrics.Recall(name='recall')
#         ]
#     )
    
#     return model


# def get_callbacks(model_path):
#     """Get training callbacks."""
#     return [
#         EarlyStopping(
#             monitor='val_auc',
#             patience=10,
#             restore_best_weights=True,
#             mode='max',
#             verbose=1
#         ),
#         ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.5,
#             patience=5,
#             min_lr=1e-7,
#             verbose=1
#         ),
#         ModelCheckpoint(
#             filepath=model_path,
#             monitor='val_auc',
#             save_best_only=True,
#             mode='max',
#             verbose=1
#         )
#     ]


# # ============================================================
# # TRAINING FUNCTION
# # ============================================================

# def train_model(csv_file, model_name, scaler_name, target_strategy='threshold'):
#     """Train model for a specific timeframe."""
#     print(f"\n{'='*60}")
#     print(f"🚀 STARTING TRAINING: {model_name}")
#     print(f"   Target Strategy: {target_strategy}")
#     print(f"{'='*60}")

#     # 1. Load and Prepare Data
#     X_data, y_data = load_and_prepare_data(csv_file, scaler_name, target_strategy)
#     if X_data is None:
#         return None

#     # 2. Create Sequences
#     print("\n✂️ Creating Sequences...")
#     X, y = create_sequences(X_data, y_data, LOOKBACK_WINDOW)
#     print(f"📊 Total sequences: {len(X)}")

#     # 3. Train/Test Split (time-series split, no shuffle!)
#     split = int(len(X) * 0.8)
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]
    
#     print(f"📊 Training: {len(X_train)} | Validation: {len(X_test)}")

#     # 4. Compute Class Weights
#     class_weights = compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(y_train),
#         y=y_train
#     )
#     class_weight_dict = dict(enumerate(class_weights))
#     print(f"⚖️ Class weights: {class_weight_dict}")

#     # 5. Build Model
#     model = build_optimized_model((X_train.shape[1], X_train.shape[2]))
#     model.summary()

#     # 6. Train
#     print(f"\n🧠 Training (max {EPOCHS} epochs, early stopping on val_auc)...")
    
#     history = model.fit(
#         X_train, y_train,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         validation_data=(X_test, y_test),
#         class_weight=class_weight_dict,
#         callbacks=get_callbacks(model_name),
#         verbose=1
#     )

#     # 7. Evaluate
#     print(f"\n📈 Final Evaluation:")
#     results = model.evaluate(X_test, y_test, verbose=0)
#     for name, value in zip(model.metrics_names, results):
#         print(f"   {name}: {value:.4f}")

#     # 8. Save History
#     history_path = model_name.replace('.keras', '_history.csv')
#     pd.DataFrame(history.history).to_csv(history_path, index=False)
#     print(f"📜 History saved to: {history_path}")

#     print(f"\n🎉 TRAINING COMPLETE: {model_name}")
    
#     return model, history


# # ============================================================
# # MAIN FUNCTION
# # ============================================================

# def main():
#     # Set seeds for reproducibility
#     np.random.seed(42)
#     tf.random.set_seed(42)
    
#     print("🔧 TensorFlow version:", tf.__version__)
#     print("🔧 GPU Available:", tf.config.list_physical_devices('GPU'))
    
#     # ============================================================
#     # CHOOSE YOUR TARGET STRATEGY HERE
#     # ============================================================
#     # Options:
#     #   'threshold'      - Predicts if price moves up by X% (recommended to start)
#     #   'triple_barrier' - Professional quant approach with TP/SL
#     #   'trend'          - Predicts trend direction
#     # ============================================================
    
#     TARGET_STRATEGY = 'threshold'  # <-- Change this to try different strategies
    
#     # --- Train 5 Minute Model ---
#     train_model(
#         csv_file="data/training/data_5m.csv",
#         model_name="models/model_5m.keras",
#         scaler_name="models/scaler_5m.pkl",
#         target_strategy=TARGET_STRATEGY
#     )

#     # --- Train 15 Minute Model ---
#     train_model(
#         csv_file="data/training/data_15m.csv",
#         model_name="models/model_15m.keras",
#         scaler_name="models/scaler_15m.pkl",
#         target_strategy=TARGET_STRATEGY
#     )

#     print("\n" + "="*60)
#     print("✅ ALL MODELS TRAINED!")
#     print("="*60)


# if __name__ == "__main__":
#     main()








# Perplkexity IC 2
# import os
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler, RobustScaler
# from sklearn.utils.class_weight import compute_class_weight
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Bidirectional, GaussianNoise
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l1_l2


# # ============================================================
# # CONFIGURATION - OPTIMIZED FOR REDUCING OVERFITTING
# # ============================================================
# LOOKBACK_WINDOW = 30  # Reduced from 60 - less complexity, faster training

# # Core features
# FEATURE_COLUMNS = [
#     'close', 'rsi', 'macd', 'macd_signal', 'williams_r', 'momentum', 'ema_20'
# ]

# # REDUCED MODEL COMPLEXITY to fight overfitting
# LSTM_UNITS_1 = 64      # Reduced from 128
# LSTM_UNITS_2 = 32      # Reduced from 64
# DENSE_UNITS = 16       # Reduced from 32

# # INCREASED REGULARIZATION
# DROPOUT_RATE = 0.4     # Increased from 0.3
# RECURRENT_DROPOUT = 0.3  # Increased from 0.2
# L1_REG = 0.001         # Added L1 regularization
# L2_REG = 0.01          # Increased from 0.001

# # Training params
# LEARNING_RATE = 0.0005  # Reduced from 0.001 for stability
# BATCH_SIZE = 128        # Increased from 64 for more stable gradients
# EPOCHS = 100            # Increased, but early stopping will control
# NOISE_STDDEV = 0.01     # Gaussian noise for input regularization

# # Target configuration
# TARGET_THRESHOLD_PCT = 0.10  # Lowered for better class balance
# TARGET_LOOKAHEAD = 8         # Increased for stronger signals


# # ============================================================
# # TARGET LABELING STRATEGIES
# # ============================================================

# def create_target_threshold(df, threshold_pct=TARGET_THRESHOLD_PCT, lookahead=TARGET_LOOKAHEAD):
#     """
#     Strategy 1: Threshold-based labeling with balanced classes
#     """
#     print(f"📊 Creating target: Threshold={threshold_pct}%, Lookahead={lookahead} candles")
    
#     df['future_close'] = df['close'].shift(-lookahead)
#     df['future_return'] = (df['future_close'] - df['close']) / df['close'] * 100
    
#     # Use absolute threshold for both directions to balance classes
#     df['target'] = (df['future_return'] > threshold_pct).astype(int)
    
#     df.dropna(subset=['target', 'future_return'], inplace=True)
#     df.drop(columns=['future_close', 'future_return'], inplace=True)
    
#     return df


# def create_target_triple_barrier(df, profit_target=0.25, stop_loss=0.15, max_holding=12):
#     """
#     Strategy 2: Triple Barrier Method (Professional Quant Approach)
#     - Optimized parameters for short timeframes
#     """
#     print(f"📊 Creating target: Triple Barrier (TP={profit_target}%, SL={stop_loss}%, MaxHold={max_holding})")
    
#     targets = []
#     closes = df['close'].values
    
#     for i in range(len(df)):
#         if i + max_holding >= len(df):
#             targets.append(np.nan)
#             continue
            
#         entry_price = closes[i]
#         tp_price = entry_price * (1 + profit_target / 100)
#         sl_price = entry_price * (1 - stop_loss / 100)
        
#         label = 0
        
#         for j in range(1, max_holding + 1):
#             future_price = closes[i + j]
            
#             if future_price >= tp_price:
#                 label = 1
#                 break
#             elif future_price <= sl_price:
#                 label = 0
#                 break
        
#         targets.append(label)
    
#     df['target'] = targets
#     df.dropna(subset=['target'], inplace=True)
#     df['target'] = df['target'].astype(int)
    
#     return df


# def create_target_volatility_adjusted(df, volatility_multiplier=1.5, lookahead=8):
#     """
#     Strategy 3: Volatility-Adjusted Threshold
#     - Adapts threshold based on recent volatility (ATR-like approach)
#     - More robust across different market conditions
#     """
#     print(f"📊 Creating target: Volatility-Adjusted (Multiplier={volatility_multiplier}, Lookahead={lookahead})")
    
#     # Calculate rolling volatility (standard deviation of returns)
#     df['returns'] = df['close'].pct_change() * 100
#     df['volatility'] = df['returns'].rolling(window=20).std()
    
#     # Dynamic threshold based on volatility
#     df['dynamic_threshold'] = df['volatility'] * volatility_multiplier
    
#     # Future return
#     df['future_close'] = df['close'].shift(-lookahead)
#     df['future_return'] = (df['future_close'] - df['close']) / df['close'] * 100
    
#     # Label: 1 if return exceeds dynamic threshold
#     df['target'] = (df['future_return'] > df['dynamic_threshold']).astype(int)
    
#     # Cleanup
#     df.drop(columns=['returns', 'volatility', 'dynamic_threshold', 'future_close', 'future_return'], inplace=True)
#     df.dropna(subset=['target'], inplace=True)
    
#     return df


# # ============================================================
# # FEATURE ENGINEERING - ENHANCED
# # ============================================================

# def add_extra_features(df):
#     """Add meaningful features with proper handling."""
#     print("🔧 Adding extra technical features...")
    
#     # Price momentum at different scales
#     df['price_change_1'] = df['close'].pct_change(1) * 100
#     df['price_change_3'] = df['close'].pct_change(3) * 100
#     df['price_change_5'] = df['close'].pct_change(5) * 100
#     df['price_change_10'] = df['close'].pct_change(10) * 100
    
#     # Volatility features
#     df['volatility_5'] = df['close'].rolling(window=5).std()
#     df['volatility_10'] = df['close'].rolling(window=10).std()
#     df['volatility_20'] = df['close'].rolling(window=20).std()
    
#     # Volatility ratio (short vs long term)
#     df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)
    
#     # RSI momentum
#     if 'rsi' in df.columns:
#         df['rsi_change'] = df['rsi'].diff(3)
#         df['rsi_ma'] = df['rsi'].rolling(window=5).mean()
#         df['rsi_deviation'] = df['rsi'] - df['rsi_ma']
    
#     # MACD features
#     if 'macd' in df.columns and 'macd_signal' in df.columns:
#         df['macd_histogram'] = df['macd'] - df['macd_signal']
#         df['macd_histogram_change'] = df['macd_histogram'].diff(1)
    
#     # Price relative to EMA
#     if 'ema_20' in df.columns:
#         df['price_vs_ema'] = (df['close'] - df['ema_20']) / df['ema_20'] * 100
#         df['ema_slope'] = df['ema_20'].diff(3) / df['ema_20'] * 100
    
#     # High-Low range (if available)
#     if 'high' in df.columns and 'low' in df.columns:
#         df['hl_range'] = (df['high'] - df['low']) / df['close'] * 100
#         df['hl_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
#     # Williams %R momentum
#     if 'williams_r' in df.columns:
#         df['williams_r_change'] = df['williams_r'].diff(3)
    
#     # Drop NaN rows
#     df.dropna(inplace=True)
    
#     return df


# # ============================================================
# # DATA LOADING AND PREPROCESSING
# # ============================================================

# def load_and_prepare_data(csv_path, scaler_save_path, target_strategy='threshold'):
#     """Load, prepare data with robust scaling."""
#     print(f"\n⏳ Loading dataset from: {csv_path}")
    
#     if not os.path.exists(csv_path):
#         print(f"❌ File not found: {csv_path}")
#         return None, None

#     df = pd.read_csv(csv_path)
#     print(f"📊 Loaded {len(df)} rows")
    
#     # Add extra features
#     df = add_extra_features(df)
    
#     # Create target based on strategy
#     if target_strategy == 'threshold':
#         df = create_target_threshold(df, threshold_pct=TARGET_THRESHOLD_PCT, lookahead=TARGET_LOOKAHEAD)
#     elif target_strategy == 'triple_barrier':
#         df = create_target_triple_barrier(df, profit_target=0.25, stop_loss=0.15, max_holding=12)
#     elif target_strategy == 'volatility':
#         df = create_target_volatility_adjusted(df, volatility_multiplier=1.5, lookahead=8)
#     else:
#         print(f"❌ Unknown target strategy: {target_strategy}")
#         return None, None
    
#     # Define feature columns
#     feature_cols = FEATURE_COLUMNS.copy()
    
#     extra_features = [
#         'price_change_1', 'price_change_3', 'price_change_5', 'price_change_10',
#         'volatility_5', 'volatility_10', 'volatility_20', 'volatility_ratio',
#         'rsi_change', 'rsi_ma', 'rsi_deviation',
#         'macd_histogram', 'macd_histogram_change',
#         'price_vs_ema', 'ema_slope',
#         'hl_range', 'hl_position',
#         'williams_r_change'
#     ]
    
#     for feat in extra_features:
#         if feat in df.columns:
#             feature_cols.append(feat)
    
#     print(f"📊 Using {len(feature_cols)} features")
    
#     # Remove any remaining NaN
#     df.dropna(inplace=True)
#     print(f"📊 After cleanup: {len(df)} rows")
    
#     if len(df) < LOOKBACK_WINDOW + 100:
#         print(f"❌ Not enough data.")
#         return None, None
    
#     # Check class balance
#     class_counts = df['target'].value_counts()
#     print(f"📊 Class distribution:\n{class_counts}")
#     class_ratio = class_counts.min() / class_counts.max()
#     print(f"📊 Class ratio: {class_ratio:.2%}")
    
#     if class_ratio < 0.25:
#         print("⚠️ Warning: Classes are imbalanced (ratio < 25%).")
    
#     # Use RobustScaler for better handling of outliers
#     data_x = df[feature_cols].values
#     data_y = df['target'].values
    
#     scaler = RobustScaler()  # More robust to outliers than MinMaxScaler
#     data_x_scaled = scaler.fit_transform(data_x)
    
#     # Clip extreme values after scaling
#     data_x_scaled = np.clip(data_x_scaled, -5, 5)
    
#     # Save scaler and feature columns
#     os.makedirs(os.path.dirname(scaler_save_path) if os.path.dirname(scaler_save_path) else '.', exist_ok=True)
#     joblib.dump({
#         'scaler': scaler,
#         'feature_columns': feature_cols
#     }, scaler_save_path)
#     print(f"✅ Scaler saved to {scaler_save_path}")
    
#     return data_x_scaled, data_y


# # ============================================================
# # SEQUENCE CREATION
# # ============================================================

# def create_sequences(data, target, lookback):
#     """Create sequences with progress indicator."""
#     X, y = [], []
#     total = len(data) - lookback
    
#     for i in range(lookback, len(data)):
#         X.append(data[i - lookback:i])
#         y.append(target[i])
        
#         if (i - lookback) % 10000 == 0 and i > lookback:
#             print(f"   Processed {i - lookback}/{total} sequences...")

#     return np.array(X), np.array(y)


# # ============================================================
# # MODEL BUILDING - OPTIMIZED TO REDUCE OVERFITTING
# # ============================================================

# def build_optimized_model(input_shape):
#     """
#     Build a simpler, more regularized LSTM model.
#     Key changes:
#     - Smaller network (fewer units)
#     - More regularization (L1+L2, higher dropout)
#     - Gaussian noise on input for data augmentation
#     - Simpler architecture
#     """
#     model = Sequential([
#         Input(shape=input_shape),
        
#         # Input noise for regularization (acts like data augmentation)
#         GaussianNoise(NOISE_STDDEV),
        
#         # First LSTM layer - Bidirectional but smaller
#         Bidirectional(LSTM(
#             units=LSTM_UNITS_1,
#             return_sequences=True,
#             kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG),
#             recurrent_regularizer=l1_l2(l1=L1_REG/10, l2=L2_REG/10),
#             bias_regularizer=l1_l2(l1=L1_REG/10, l2=L2_REG/10),
#             dropout=DROPOUT_RATE,
#             recurrent_dropout=RECURRENT_DROPOUT
#         )),
#         BatchNormalization(),

#         # Second LSTM layer
#         LSTM(
#             units=LSTM_UNITS_2,
#             return_sequences=False,
#             kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG),
#             recurrent_regularizer=l1_l2(l1=L1_REG/10, l2=L2_REG/10),
#             dropout=DROPOUT_RATE,
#             recurrent_dropout=RECURRENT_DROPOUT
#         ),
#         BatchNormalization(),
#         Dropout(DROPOUT_RATE),

#         # Dense layer
#         Dense(
#             units=DENSE_UNITS, 
#             activation='relu', 
#             kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG)
#         ),
#         BatchNormalization(),
#         Dropout(DROPOUT_RATE),

#         # Output layer
#         Dense(units=1, activation='sigmoid')
#     ])

#     optimizer = Adam(learning_rate=LEARNING_RATE)
    
#     model.compile(
#         optimizer=optimizer,
#         loss='binary_crossentropy',
#         metrics=[
#             'accuracy',
#             tf.keras.metrics.AUC(name='auc'),
#             tf.keras.metrics.Precision(name='precision'),
#             tf.keras.metrics.Recall(name='recall')
#         ]
#     )
    
#     return model


# def get_callbacks(model_path):
#     """Training callbacks with more patience for regularized models."""
#     return [
#         EarlyStopping(
#             monitor='val_auc',
#             patience=15,  # Increased patience
#             restore_best_weights=True,
#             mode='max',
#             verbose=1,
#             min_delta=0.001  # Minimum improvement required
#         ),
#         ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.5,
#             patience=7,  # Increased patience
#             min_lr=1e-7,
#             verbose=1
#         ),
#         ModelCheckpoint(
#             filepath=model_path,
#             monitor='val_auc',
#             save_best_only=True,
#             mode='max',
#             verbose=1
#         )
#     ]


# # ============================================================
# # TRAINING FUNCTION
# # ============================================================

# def train_model(csv_file, model_name, scaler_name, target_strategy='threshold'):
#     """Train model with all optimizations."""
#     print(f"\n{'='*60}")
#     print(f"🚀 STARTING TRAINING: {model_name}")
#     print(f"   Target Strategy: {target_strategy}")
#     print(f"   Lookback Window: {LOOKBACK_WINDOW}")
#     print(f"   Model: LSTM({LSTM_UNITS_1}) -> LSTM({LSTM_UNITS_2}) -> Dense({DENSE_UNITS})")
#     print(f"   Regularization: L1={L1_REG}, L2={L2_REG}, Dropout={DROPOUT_RATE}")
#     print(f"{'='*60}")

#     # 1. Load and Prepare Data
#     X_data, y_data = load_and_prepare_data(csv_file, scaler_name, target_strategy)
#     if X_data is None:
#         return None

#     # 2. Create Sequences
#     print("\n✂️ Creating Sequences...")
#     X, y = create_sequences(X_data, y_data, LOOKBACK_WINDOW)
#     print(f"📊 Total sequences: {len(X)}")

#     # 3. Train/Validation/Test Split
#     # Use 70% train, 15% validation, 15% test
#     train_split = int(len(X) * 0.70)
#     val_split = int(len(X) * 0.85)
    
#     X_train = X[:train_split]
#     y_train = y[:train_split]
#     X_val = X[train_split:val_split]
#     y_val = y[train_split:val_split]
#     X_test = X[val_split:]
#     y_test = y[val_split:]
    
#     print(f"📊 Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

#     # 4. Compute Class Weights
#     class_weights = compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(y_train),
#         y=y_train
#     )
#     class_weight_dict = dict(enumerate(class_weights))
#     print(f"⚖️ Class weights: {class_weight_dict}")

#     # 5. Build Model
#     model = build_optimized_model((X_train.shape[1], X_train.shape[2]))
#     model.summary()

#     # 6. Train
#     print(f"\n🧠 Training (max {EPOCHS} epochs, early stopping on val_auc)...")
    
#     history = model.fit(
#         X_train, y_train,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         validation_data=(X_val, y_val),
#         class_weight=class_weight_dict,
#         callbacks=get_callbacks(model_name),
#         verbose=1
#     )

#     # 7. Evaluate on TEST set (unseen data)
#     print(f"\n📈 Final Evaluation on TEST Set (Unseen Data):")
#     results = model.evaluate(X_test, y_test, verbose=0)
#     for name, value in zip(model.metrics_names, results):
#         print(f"   {name}: {value:.4f}")
    
#     # 8. Calculate additional metrics
#     y_pred_proba = model.predict(X_test, verbose=0)
#     y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
#     # Confusion matrix style metrics
#     true_positives = np.sum((y_pred == 1) & (y_test == 1))
#     false_positives = np.sum((y_pred == 1) & (y_test == 0))
#     true_negatives = np.sum((y_pred == 0) & (y_test == 0))
#     false_negatives = np.sum((y_pred == 0) & (y_test == 1))
    
#     print(f"\n📊 Confusion Matrix Breakdown:")
#     print(f"   True Positives:  {true_positives}")
#     print(f"   False Positives: {false_positives}")
#     print(f"   True Negatives:  {true_negatives}")
#     print(f"   False Negatives: {false_negatives}")
    
#     if (true_positives + false_positives) > 0:
#         precision = true_positives / (true_positives + false_positives)
#         print(f"   Precision (when model says BUY): {precision:.2%}")
    
#     if (true_positives + false_negatives) > 0:
#         recall = true_positives / (true_positives + false_negatives)
#         print(f"   Recall (% of actual UPs caught): {recall:.2%}")

#     # 9. Save History
#     history_path = model_name.replace('.keras', '_history.csv')
#     pd.DataFrame(history.history).to_csv(history_path, index=False)
#     print(f"📜 History saved to: {history_path}")

#     print(f"\n🎉 TRAINING COMPLETE: {model_name}")
    
#     return model, history


# # ============================================================
# # MAIN FUNCTION
# # ============================================================

# def main():
#     # Set seeds for reproducibility
#     np.random.seed(42)
#     tf.random.set_seed(42)
    
#     print("="*60)
#     print("🔧 OPTIMIZED LSTM TRAINING SCRIPT")
#     print("="*60)
#     print(f"🔧 TensorFlow version: {tf.__version__}")
#     print(f"🔧 GPU Available: {tf.config.list_physical_devices('GPU')}")
#     print(f"🔧 Lookback Window: {LOOKBACK_WINDOW}")
#     print(f"🔧 Target Threshold: {TARGET_THRESHOLD_PCT}%")
#     print(f"🔧 Target Lookahead: {TARGET_LOOKAHEAD} candles")
    
#     # ============================================================
#     # CHOOSE YOUR TARGET STRATEGY
#     # ============================================================
#     # Options:
#     #   'threshold'     - Simple percentage threshold (recommended)
#     #   'triple_barrier' - Professional TP/SL approach
#     #   'volatility'    - Volatility-adjusted threshold
#     # ============================================================
    
#     TARGET_STRATEGY = 'threshold'  # Change to try others
    
#     print(f"🔧 Target Strategy: {TARGET_STRATEGY}")
#     print("="*60)

#     # --- Train 5 Minute Model ---
#     train_model(
#         csv_file="data/training/data_5m.csv",
#         model_name="models/model_5m.keras",
#         scaler_name="models/scaler_5m.pkl",
#         target_strategy=TARGET_STRATEGY
#     )

#     # --- Train 15 Minute Model ---
#     train_model(
#         csv_file="data/training/data_15m.csv",
#         model_name="models/model_15m.keras",
#         scaler_name="models/scaler_15m.pkl",
#         target_strategy=TARGET_STRATEGY
#     )

#     print("\n" + "="*60)
#     print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
#     print("="*60)
#     print("\n📋 NEXT STEPS:")
#     print("   1. Check val_auc vs train_auc gap (should be smaller now)")
#     print("   2. Look at Test Set metrics (separate from validation)")
#     print("   3. Try 'triple_barrier' strategy if threshold doesn't work")
#     print("   4. Consider longer timeframe data (1h, 4h) for cleaner signals")


# if __name__ == "__main__":
#     main()





# Perplexity IC 3
# import os
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import RobustScaler
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, confusion_matrix
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import (
#     LSTM, Dense, Dropout, Input, BatchNormalization, 
#     Bidirectional, GaussianNoise, Attention, Concatenate,
#     GlobalAveragePooling1D, Multiply, Permute, RepeatVector,
#     Lambda, Flatten
# )
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l1_l2


# # ============================================================
# # CONFIGURATION - PRODUCTION OPTIMIZED
# # ============================================================
# LOOKBACK_WINDOW = 30

# FEATURE_COLUMNS = [
#     'close', 'rsi', 'macd', 'macd_signal', 'williams_r', 'momentum', 'ema_20'
# ]

# # Model Architecture
# LSTM_UNITS_1 = 48
# LSTM_UNITS_2 = 24
# DENSE_UNITS = 16

# # Regularization
# DROPOUT_RATE = 0.5
# RECURRENT_DROPOUT = 0.3
# L1_REG = 0.002
# L2_REG = 0.02
# NOISE_STDDEV = 0.02

# # Training
# LEARNING_RATE = 0.0003
# BATCH_SIZE = 256
# EPOCHS = 150

# # Triple Barrier Parameters (OPTIMIZED)
# PROFIT_TARGET_PCT = 0.20    # Take profit at 0.20%
# STOP_LOSS_PCT = 0.12        # Stop loss at 0.12%
# MAX_HOLDING_CANDLES = 15    # Max hold 15 candles

# # Prediction Threshold (for live trading)
# CONFIDENCE_THRESHOLD = 0.60  # Only trade when probability > 60%


# # ============================================================
# # TRIPLE BARRIER TARGET - PROFESSIONAL QUANT METHOD
# # ============================================================

# def create_triple_barrier_target(df, profit_target=PROFIT_TARGET_PCT, 
#                                   stop_loss=STOP_LOSS_PCT, 
#                                   max_holding=MAX_HOLDING_CANDLES):
#     """
#     Triple Barrier Method:
#     - Upper barrier: Take Profit
#     - Lower barrier: Stop Loss  
#     - Vertical barrier: Max holding period
    
#     Label = 1 if price hits Take Profit FIRST
#     Label = 0 if price hits Stop Loss first OR timeout
#     """
#     print(f"📊 Creating Triple Barrier Target:")
#     print(f"   Take Profit: {profit_target}%")
#     print(f"   Stop Loss: {stop_loss}%")
#     print(f"   Max Holding: {max_holding} candles")
    
#     targets = []
#     touch_types = []  # For analysis
#     closes = df['close'].values
    
#     for i in range(len(df)):
#         if i + max_holding >= len(df):
#             targets.append(np.nan)
#             touch_types.append('incomplete')
#             continue
            
#         entry_price = closes[i]
#         tp_price = entry_price * (1 + profit_target / 100)
#         sl_price = entry_price * (1 - stop_loss / 100)
        
#         label = 0
#         touch = 'timeout'
        
#         for j in range(1, max_holding + 1):
#             future_price = closes[i + j]
            
#             # Check Take Profit first (bullish bias)
#             if future_price >= tp_price:
#                 label = 1
#                 touch = 'tp_hit'
#                 break
#             # Then check Stop Loss
#             elif future_price <= sl_price:
#                 label = 0
#                 touch = 'sl_hit'
#                 break
        
#         targets.append(label)
#         touch_types.append(touch)
    
#     df['target'] = targets
#     df['touch_type'] = touch_types
    
#     # Analyze touch distribution
#     touch_counts = df['touch_type'].value_counts()
#     print(f"\n📊 Barrier Touch Analysis:")
#     for touch, count in touch_counts.items():
#         pct = count / len(df) * 100
#         print(f"   {touch}: {count} ({pct:.1f}%)")
    
#     df.drop(columns=['touch_type'], inplace=True)
#     df.dropna(subset=['target'], inplace=True)
#     df['target'] = df['target'].astype(int)
    
#     return df


# # ============================================================
# # FEATURE ENGINEERING - ENHANCED
# # ============================================================

# def add_features(df):
#     """Add carefully selected features."""
#     print("🔧 Adding technical features...")
    
#     # Price momentum
#     df['returns_1'] = df['close'].pct_change(1) * 100
#     df['returns_3'] = df['close'].pct_change(3) * 100
#     df['returns_5'] = df['close'].pct_change(5) * 100
#     df['returns_10'] = df['close'].pct_change(10) * 100
    
#     # Volatility
#     df['volatility_5'] = df['returns_1'].rolling(5).std()
#     df['volatility_10'] = df['returns_1'].rolling(10).std()
#     df['volatility_20'] = df['returns_1'].rolling(20).std()
    
#     # Volatility regime
#     df['vol_regime'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)
    
#     # RSI features
#     if 'rsi' in df.columns:
#         df['rsi_sma'] = df['rsi'].rolling(5).mean()
#         df['rsi_std'] = df['rsi'].rolling(10).std()
#         df['rsi_zscore'] = (df['rsi'] - df['rsi_sma']) / (df['rsi_std'] + 1e-8)
    
#     # MACD features
#     if 'macd' in df.columns and 'macd_signal' in df.columns:
#         df['macd_hist'] = df['macd'] - df['macd_signal']
#         df['macd_hist_change'] = df['macd_hist'].diff()
#         df['macd_cross'] = np.sign(df['macd_hist']) != np.sign(df['macd_hist'].shift(1))
#         df['macd_cross'] = df['macd_cross'].astype(int)
    
#     # Price vs EMA
#     if 'ema_20' in df.columns:
#         df['price_ema_ratio'] = (df['close'] / df['ema_20'] - 1) * 100
#         df['ema_slope'] = df['ema_20'].pct_change(3) * 100
    
#     # High-Low features
#     if 'high' in df.columns and 'low' in df.columns:
#         df['hl_range'] = (df['high'] - df['low']) / df['close'] * 100
#         df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
#     # Williams %R features
#     if 'williams_r' in df.columns:
#         df['willr_sma'] = df['williams_r'].rolling(5).mean()
#         df['willr_extreme'] = ((df['williams_r'] < -80) | (df['williams_r'] > -20)).astype(int)
    
#     # Momentum features
#     if 'momentum' in df.columns:
#         df['momentum_sma'] = df['momentum'].rolling(5).mean()
#         df['momentum_accel'] = df['momentum'].diff()
    
#     df.dropna(inplace=True)
#     return df


# # ============================================================
# # DATA LOADING
# # ============================================================

# def load_and_prepare_data(csv_path, scaler_save_path):
#     """Load and prepare data with triple barrier labeling."""
#     print(f"\n⏳ Loading: {csv_path}")
    
#     if not os.path.exists(csv_path):
#         print(f"❌ File not found: {csv_path}")
#         return None, None

#     df = pd.read_csv(csv_path)
#     print(f"📊 Loaded {len(df)} rows")
    
#     # Add features
#     df = add_features(df)
    
#     # Create triple barrier target
#     df = create_triple_barrier_target(df)
    
#     # Get all feature columns
#     feature_cols = FEATURE_COLUMNS.copy()
#     extra_features = [
#         'returns_1', 'returns_3', 'returns_5', 'returns_10',
#         'volatility_5', 'volatility_10', 'volatility_20', 'vol_regime',
#         'rsi_sma', 'rsi_std', 'rsi_zscore',
#         'macd_hist', 'macd_hist_change', 'macd_cross',
#         'price_ema_ratio', 'ema_slope',
#         'hl_range', 'close_position',
#         'willr_sma', 'willr_extreme',
#         'momentum_sma', 'momentum_accel'
#     ]
    
#     for feat in extra_features:
#         if feat in df.columns:
#             feature_cols.append(feat)
    
#     print(f"📊 Using {len(feature_cols)} features")
    
#     # Clean data
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df.dropna(inplace=True)
#     print(f"📊 After cleanup: {len(df)} rows")
    
#     # Class distribution
#     class_counts = df['target'].value_counts()
#     print(f"\n📊 Class Distribution:")
#     print(f"   Class 0 (No Trade/Loss): {class_counts.get(0, 0)}")
#     print(f"   Class 1 (Profitable):    {class_counts.get(1, 0)}")
#     print(f"   Ratio: {class_counts.get(1, 0) / len(df) * 100:.1f}%")
    
#     # Scale features
#     X = df[feature_cols].values
#     y = df['target'].values
    
#     scaler = RobustScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_scaled = np.clip(X_scaled, -5, 5)
    
#     # Save scaler and feature info
#     os.makedirs(os.path.dirname(scaler_save_path) if os.path.dirname(scaler_save_path) else '.', exist_ok=True)
#     joblib.dump({
#         'scaler': scaler,
#         'feature_columns': feature_cols,
#         'triple_barrier_params': {
#             'profit_target': PROFIT_TARGET_PCT,
#             'stop_loss': STOP_LOSS_PCT,
#             'max_holding': MAX_HOLDING_CANDLES
#         }
#     }, scaler_save_path)
#     print(f"✅ Scaler saved: {scaler_save_path}")
    
#     return X_scaled, y


# # ============================================================
# # SEQUENCE CREATION
# # ============================================================

# def create_sequences(X, y, lookback):
#     """Create sequences for LSTM."""
#     Xs, ys = [], []
#     for i in range(lookback, len(X)):
#         Xs.append(X[i-lookback:i])
#         ys.append(y[i])
#     return np.array(Xs), np.array(ys)


# # ============================================================
# # MODEL ARCHITECTURE - WITH SIMPLE ATTENTION
# # ============================================================

# def build_model(input_shape):
#     """
#     Compact LSTM with attention-like mechanism.
#     Smaller, more regularized, faster to train.
#     """
#     inputs = Input(shape=input_shape)
    
#     # Noise for regularization
#     x = GaussianNoise(NOISE_STDDEV)(inputs)
    
#     # Bidirectional LSTM
#     x = Bidirectional(LSTM(
#         units=LSTM_UNITS_1,
#         return_sequences=True,
#         kernel_regularizer=l1_l2(L1_REG, L2_REG),
#         recurrent_regularizer=l1_l2(L1_REG/10, L2_REG/10),
#         dropout=DROPOUT_RATE,
#         recurrent_dropout=RECURRENT_DROPOUT
#     ))(x)
#     x = BatchNormalization()(x)
    
#     # Second LSTM
#     x = LSTM(
#         units=LSTM_UNITS_2,
#         return_sequences=True,
#         kernel_regularizer=l1_l2(L1_REG, L2_REG),
#         dropout=DROPOUT_RATE,
#         recurrent_dropout=RECURRENT_DROPOUT
#     )(x)
#     x = BatchNormalization()(x)
    
#     # Simple attention: learn which timesteps are important
#     attention = Dense(1, activation='tanh')(x)
#     attention = Flatten()(attention)
#     attention = tf.keras.layers.Activation('softmax')(attention)
#     attention = RepeatVector(LSTM_UNITS_2)(attention)
#     attention = Permute([2, 1])(attention)
    
#     # Apply attention
#     x = Multiply()([x, attention])
#     x = Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
    
#     # Dense layers
#     x = Dense(DENSE_UNITS, activation='relu', kernel_regularizer=l1_l2(L1_REG, L2_REG))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(DROPOUT_RATE)(x)
    
#     # Output
#     outputs = Dense(1, activation='sigmoid')(x)
    
#     model = Model(inputs, outputs)
    
#     model.compile(
#         optimizer=Adam(learning_rate=LEARNING_RATE),
#         loss='binary_crossentropy',
#         metrics=[
#             'accuracy',
#             tf.keras.metrics.AUC(name='auc'),
#             tf.keras.metrics.Precision(name='precision'),
#             tf.keras.metrics.Recall(name='recall')
#         ]
#     )
    
#     return model


# # ============================================================
# # CALLBACKS
# # ============================================================

# def get_callbacks(model_path):
#     return [
#         EarlyStopping(
#             monitor='val_auc',
#             patience=20,
#             restore_best_weights=True,
#             mode='max',
#             verbose=1,
#             min_delta=0.001
#         ),
#         ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.5,
#             patience=8,
#             min_lr=1e-7,
#             verbose=1
#         ),
#         ModelCheckpoint(
#             filepath=model_path,
#             monitor='val_auc',
#             save_best_only=True,
#             mode='max',
#             verbose=1
#         )
#     ]


# # ============================================================
# # TRAINING FUNCTION
# # ============================================================

# def train_model(csv_file, model_name, scaler_name):
#     """Train model with triple barrier method."""
#     print(f"\n{'='*60}")
#     print(f"🚀 TRAINING: {model_name}")
#     print(f"   Method: Triple Barrier")
#     print(f"   TP: {PROFIT_TARGET_PCT}% | SL: {STOP_LOSS_PCT}% | MaxHold: {MAX_HOLDING_CANDLES}")
#     print(f"{'='*60}")

#     # Load data
#     X, y = load_and_prepare_data(csv_file, scaler_name)
#     if X is None:
#         return None

#     # Create sequences
#     print("\n✂️ Creating sequences...")
#     X_seq, y_seq = create_sequences(X, y, LOOKBACK_WINDOW)
#     print(f"📊 Total sequences: {len(X_seq)}")

#     # Split: 70% train, 15% val, 15% test
#     train_end = int(len(X_seq) * 0.70)
#     val_end = int(len(X_seq) * 0.85)
    
#     X_train, y_train = X_seq[:train_end], y_seq[:train_end]
#     X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
#     X_test, y_test = X_seq[val_end:], y_seq[val_end:]
    
#     print(f"📊 Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

#     # Class weights
#     class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#     class_weight_dict = dict(enumerate(class_weights))
#     print(f"⚖️ Class weights: {class_weight_dict}")

#     # Build model
#     model = build_model((X_train.shape[1], X_train.shape[2]))
#     model.summary()

#     # Train
#     print(f"\n🧠 Training (max {EPOCHS} epochs)...")
#     history = model.fit(
#         X_train, y_train,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         validation_data=(X_val, y_val),
#         class_weight=class_weight_dict,
#         callbacks=get_callbacks(model_name),
#         verbose=1
#     )

#     # ============================================================
#     # COMPREHENSIVE EVALUATION
#     # ============================================================
#     print(f"\n{'='*60}")
#     print("📈 EVALUATION ON TEST SET (UNSEEN DATA)")
#     print(f"{'='*60}")
    
#     # Basic metrics
#     results = model.evaluate(X_test, y_test, verbose=0)
#     for name, value in zip(model.metrics_names, results):
#         print(f"   {name}: {value:.4f}")
    
#     # Predictions
#     y_pred_proba = model.predict(X_test, verbose=0).flatten()
#     y_pred_default = (y_pred_proba > 0.5).astype(int)
#     y_pred_confident = (y_pred_proba > CONFIDENCE_THRESHOLD).astype(int)
    
#     # Confusion Matrix - Default (0.5 threshold)
#     print(f"\n📊 Confusion Matrix (threshold=0.5):")
#     cm = confusion_matrix(y_test, y_pred_default)
#     tn, fp, fn, tp = cm.ravel()
#     print(f"   True Negatives:  {tn}")
#     print(f"   False Positives: {fp}")
#     print(f"   False Negatives: {fn}")
#     print(f"   True Positives:  {tp}")
    
#     precision_05 = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall_05 = tp / (tp + fn) if (tp + fn) > 0 else 0
#     print(f"   Precision: {precision_05:.2%}")
#     print(f"   Recall: {recall_05:.2%}")
    
#     # High Confidence Predictions
#     print(f"\n📊 High Confidence Signals (threshold={CONFIDENCE_THRESHOLD}):")
#     confident_mask = y_pred_proba > CONFIDENCE_THRESHOLD
#     n_confident = confident_mask.sum()
    
#     if n_confident > 0:
#         confident_correct = (y_pred_confident[confident_mask] == y_test[confident_mask]).sum()
#         confident_precision = confident_correct / n_confident
#         print(f"   Total signals: {n_confident}")
#         print(f"   Correct: {confident_correct}")
#         print(f"   Precision: {confident_precision:.2%}")
        
#         # Win rate analysis
#         if confident_precision > 0.5:
#             print(f"   ✅ PROFITABLE: Win rate > 50%")
#         else:
#             print(f"   ⚠️ NOT PROFITABLE: Win rate < 50%")
#     else:
#         print(f"   No signals above {CONFIDENCE_THRESHOLD} threshold")
    
#     # Probability distribution
#     print(f"\n📊 Prediction Probability Distribution:")
#     for threshold in [0.4, 0.5, 0.6, 0.7, 0.8]:
#         count = (y_pred_proba > threshold).sum()
#         if count > 0:
#             correct = ((y_pred_proba > threshold) & (y_test == 1)).sum()
#             acc = correct / count
#             print(f"   >{threshold}: {count} signals, {acc:.1%} accuracy")
    
#     # Save history
#     history_path = model_name.replace('.keras', '_history.csv')
#     pd.DataFrame(history.history).to_csv(history_path, index=False)
#     print(f"\n📜 History saved: {history_path}")
    
#     print(f"\n🎉 TRAINING COMPLETE: {model_name}")
    
#     return model, history


# # ============================================================
# # MAIN
# # ============================================================

# def main():
#     np.random.seed(42)
#     tf.random.set_seed(42)
    
#     print("="*60)
#     print("🚀 PRODUCTION LSTM TRAINING - TRIPLE BARRIER METHOD")
#     print("="*60)
#     print(f"TensorFlow: {tf.__version__}")
#     print(f"GPU: {tf.config.list_physical_devices('GPU')}")
#     print(f"\nTriple Barrier Settings:")
#     print(f"   Take Profit: {PROFIT_TARGET_PCT}%")
#     print(f"   Stop Loss: {STOP_LOSS_PCT}%")
#     print(f"   Max Holding: {MAX_HOLDING_CANDLES} candles")
#     print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD}")
#     print("="*60)

#     # Train 5m model
#     train_model(
#         csv_file="data/training/data_5m.csv",
#         model_name="models/model_5m.keras",
#         scaler_name="models/scaler_5m.pkl"
#     )

#     # Train 15m model
#     train_model(
#         csv_file="data/training/data_15m.csv",
#         model_name="models/model_15m.keras",
#         scaler_name="models/scaler_15m.pkl"
#     )

#     print("\n" + "="*60)
#     print("✅ ALL MODELS TRAINED!")
#     print("="*60)
#     print("\n📋 TRADING RULES:")
#     print(f"   1. Only trade when probability > {CONFIDENCE_THRESHOLD}")
#     print(f"   2. Set Take Profit at {PROFIT_TARGET_PCT}%")
#     print(f"   3. Set Stop Loss at {STOP_LOSS_PCT}%")
#     print(f"   4. Exit after {MAX_HOLDING_CANDLES} candles if no TP/SL hit")
#     print("="*60)


# if __name__ == "__main__":
#     main()







# New Gemini 

# import os
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import RobustScaler
# from sklearn.utils.class_weight import compute_class_weight
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import GRU, Dense, Dropout, Input, BatchNormalization, GaussianNoise
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2

# # ============================================================
# # ⚙️ CONFIGURATION (OPTIMIZED FOR 5M/15M)
# # ============================================================
# LOOKBACK_WINDOW = 30  
# BATCH_SIZE = 64       
# EPOCHS = 100          
# LEARNING_RATE = 0.0005 

# # STRICT REGULARIZATION (To stop Ratta-fication/Overfitting)
# DROPOUT_RATE = 0.5     # High dropout to force learning
# L2_REG = 0.01          # Strong penalty for complex weights
# NOISE_STDDEV = 0.05    # Add random noise to confuse the model slightly (makes it robust)

# # ============================================================
# # 1. SMART FEATURE ENGINEERING
# # ============================================================
# def add_technical_features(df):
#     """Calculates only the MOST POWERFUL indicators."""
#     print("🔧 Generating Alpha Features...")
    
#     # 1. Returns (The most honest signal)
#     df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
#     # 2. RSI (Momentum)
#     delta = df['close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs = gain / loss
#     df['rsi'] = 100 - (100 / (1 + rs))
    
#     # 3. MACD (Trend)
#     ema12 = df['close'].ewm(span=12, adjust=False).mean()
#     ema26 = df['close'].ewm(span=26, adjust=False).mean()
#     df['macd'] = ema12 - ema26
    
#     # 4. Bollinger Bands (Volatility)
#     sma20 = df['close'].rolling(window=20).mean()
#     std20 = df['close'].rolling(window=20).std()
#     df['bb_upper'] = (sma20 + 2 * std20 - df['close']) / df['close'] # Normalized
#     df['bb_lower'] = (df['close'] - (sma20 - 2 * std20)) / df['close'] # Normalized
    
#     # 5. Volume Trend
#     if 'volume' in df.columns:
#         df['vol_ma'] = df['volume'] / df['volume'].rolling(window=20).mean()
#     else:
#         df['vol_ma'] = 0
        
#     # 6. Distance from EMA (Trend Strength)
#     ema50 = df['close'].ewm(span=50, adjust=False).mean()
#     df['dist_ema50'] = (df['close'] - ema50) / ema50

#     return df

# # Features to actually use (Only the best ones)
# FEATURES_TO_USE = [
#     'log_ret', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'vol_ma', 'dist_ema50'
# ]

# # ============================================================
# # 2. SMART DATA LOADER
# # ============================================================
# def load_and_prep_data(csv_path, scaler_save_path):
#     print(f"\n⏳ Loading: {csv_path}")
#     if not os.path.exists(csv_path):
#         print("❌ File not found!")
#         return None, None
        
#     df = pd.read_csv(csv_path)
    
#     # Add Features
#     df = add_technical_features(df)
    
#     # --- VOLATILITY BASED TARGET (The "Quant" Way) ---
#     # Instead of fixed %, we target 1.5x the current volatility
#     # This implies: If market is moving fast, expect big move. If slow, small move.
#     volatility = df['log_ret'].rolling(window=20).std()
#     future_return = df['close'].shift(-8) / df['close'] - 1 # 8 candles ahead
    
#     # Target 1: If Future Return > 1.0 * Volatility (BUY)
#     df['target'] = (future_return > (volatility * 1.0)).astype(int)
    
#     # Clean NaN
#     df.dropna(inplace=True)
    
#     print(f"📊 Features used: {len(FEATURES_TO_USE)}")
#     print(f"📊 Class Balance: {df['target'].value_counts().to_dict()}")
    
#     # Scaling
#     data_x = df[FEATURES_TO_USE].values
#     data_y = df['target'].values
    
#     scaler = RobustScaler()
#     data_x_scaled = scaler.fit_transform(data_x)
    
#     # Save Scaler
#     os.makedirs("models", exist_ok=True)
#     joblib.dump({'scaler': scaler, 'features': FEATURES_TO_USE}, scaler_save_path)
    
#     return data_x_scaled, data_y

# # ============================================================
# # 3. THE GRU ARCHITECTURE (Lighter & Faster)
# # ============================================================
# def build_gru_model(input_shape):
#     model = Sequential([
#         Input(shape=input_shape),
        
#         # Noise injection (prevents overfitting on small timeframes)
#         GaussianNoise(NOISE_STDDEV),
        
#         # Layer 1: GRU (Simpler than LSTM)
#         GRU(64, return_sequences=True, 
#             kernel_regularizer=l2(L2_REG), 
#             dropout=DROPOUT_RATE),
        
#         BatchNormalization(),
        
#         # Layer 2: GRU
#         GRU(32, return_sequences=False, 
#             kernel_regularizer=l2(L2_REG), 
#             dropout=DROPOUT_RATE),
            
#         BatchNormalization(),
        
#         # Dense Head
#         Dense(16, activation='relu', kernel_regularizer=l2(L2_REG)),
#         Dropout(0.3),
        
#         # Output
#         Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(
#         optimizer=Adam(learning_rate=LEARNING_RATE),
#         loss='binary_crossentropy',
#         metrics=['accuracy', 'AUC']
#     )
#     return model

# # ============================================================
# # 4. TRAINING LOOP
# # ============================================================
# def train_routine(csv_file, model_name, scaler_name):
#     print(f"\n🚀 Training {model_name}...")
    
#     # 1. Get Data
#     X, y = load_and_prep_data(csv_file, scaler_name)
#     if X is None: return
    
#     # 2. Make Sequences
#     X_seq, y_seq = [], []
#     for i in range(LOOKBACK_WINDOW, len(X)):
#         X_seq.append(X[i-LOOKBACK_WINDOW:i])
#         y_seq.append(y[i])
#     X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    
#     # 3. Split
#     split = int(len(X_seq) * 0.85) # 85% Train, 15% Val
#     X_train, y_train = X_seq[:split], y_seq[:split]
#     X_val, y_val = X_seq[split:], y_seq[split:]
    
#     # 4. Class Weights (Handle imbalance)
#     cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#     cw_dict = dict(enumerate(cw))
    
#     # 5. Build & Train
#     model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    
#     callbacks = [
#         EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
#     ]
    
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         class_weight=cw_dict,
#         callbacks=callbacks,
#         verbose=1
#     )
    
#     # 6. Save
#     model.save(model_name)
#     print(f"🎉 Saved {model_name}")
    
#     # 7. Final Score
#     loss, acc, auc = model.evaluate(X_val, y_val, verbose=0)
#     print(f"🏆 Final Result -> Accuracy: {acc:.2%} | AUC: {auc:.4f}")

# def main():
#     # Train 5M
#     train_routine("data/training/data_5m.csv", "models/model_5m.keras", "models/scaler_5m.pkl")
#     # Train 15M
#     train_routine("data/training/data_15m.csv", "models/model_15m.keras", "models/scaler_15m.pkl")
    
#     print("\n✅ DONE! All models upgraded to GRU.")

# if __name__ == "__main__":
#     main()







# New Gemini Triple Barrier

# import os
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# # ============================================================
# # ⚙️ CONFIGURATION
# # ============================================================
# BATCH_SIZE = 32
# EPOCHS = 50
# LEARNING_RATE = 0.001

# # ============================================================
# # 1. FEATURE ENGINEERING
# # ============================================================
# def add_features(df):
#     """Calculates purely technical features"""
#     print("🔧 Generating Features...")
    
#     # RSI
#     delta = df['close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs = gain / loss
#     df['rsi'] = 100 - (100 / (1 + rs))
    
#     # MACD
#     ema12 = df['close'].ewm(span=12).mean()
#     ema26 = df['close'].ewm(span=26).mean()
#     df['macd'] = ema12 - ema26
#     df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
#     # Bollinger Bands
#     sma20 = df['close'].rolling(window=20).mean()
#     std20 = df['close'].rolling(window=20).std()
#     df['bb_width'] = (sma20 + 2 * std20 - (sma20 - 2 * std20)) / df['close']
    
#     # SMA Crossover
#     df['sma_50'] = df['close'].rolling(window=50).mean()
#     df['dist_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    
#     # Target: 1 if Price > SMA_20 (Uptrend), 0 if Price < SMA_20 (Downtrend) 
#     # We are teaching the AI to recognize "Trends", not predict the future.
#     df['target'] = (df['close'] > sma20).astype(int)
    
#     return df.dropna()

# FEATURES = ['rsi', 'macd', 'macd_signal', 'bb_width', 'dist_sma50']

# # ============================================================
# # 2. SIMPLE DATA LOADER
# # ============================================================
# def load_data(csv_path, scaler_path):
#     df = pd.read_csv(csv_path)
#     df = add_features(df)
    
#     X = df[FEATURES].values
#     y = df['target'].values
    
#     # Scale
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     joblib.dump({'scaler': scaler, 'features': FEATURES}, scaler_path)
#     return X_scaled, y
 
# # ============================================================
# # 3. SIMPLE MODEL (Dense Network)
# # ============================================================
# def build_model(input_dim):
#     model = Sequential([
#         Input(shape=(input_dim,)),
#         Dense(64, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.2),
#         Dense(32, activation='relu'),
#         Dropout(0.2),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
#                   loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # ============================================================
# # 4. TRAINING
# # ============================================================
# def train(csv_file, model_name, scaler_name):
#     print(f"\n🚀 Training {model_name}...")
    
#     X, y = load_data(csv_file, scaler_name)
    
#     # Split
#     split = int(len(X) * 0.8)
#     X_train, y_train = X[:split], y[:split]
#     X_val, y_val = X[split:], y[split:]
    
#     model = build_model(X_train.shape[1])
    
#     model.fit(X_train, y_train, validation_data=(X_val, y_val),
#               epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
#     model.save(model_name)
#     loss, acc = model.evaluate(X_val, y_val, verbose=0)
#     print(f"🏆 Accuracy: {acc:.2%}")

# if __name__ == "__main__":
#     train("data/training/data_5m.csv", "models/model_5m.keras", "models/scaler_5m.pkl")
#     train("data/training/data_15m.csv", "models/model_15m.keras", "models/scaler_15m.pkl")




# Accuracy Model with Triple Barrier Method
# # ============================================================# Main # ============================================================

# import os
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# # ============================================================
# # ⚙️ CONFIGURATION
# # ============================================================
# BATCH_SIZE = 32
# EPOCHS = 50
# LEARNING_RATE = 0.001

# # ============================================================
# # 1. FEATURE ENGINEERING
# # ============================================================
# def add_features(df):
#     """Calculates purely technical features"""
#     print("🔧 Generating Features...")
    
#     # RSI
#     delta = df['close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs = gain / loss
#     df['rsi'] = 100 - (100 / (1 + rs))
    
#     # MACD
#     ema12 = df['close'].ewm(span=12).mean()
#     ema26 = df['close'].ewm(span=26).mean()
#     df['macd'] = ema12 - ema26
#     df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
#     # Bollinger Bands
#     sma20 = df['close'].rolling(window=20).mean()
#     std20 = df['close'].rolling(window=20).std()
#     df['bb_width'] = (sma20 + 2 * std20 - (sma20 - 2 * std20)) / df['close']
    
#     # SMA Crossover
#     df['sma_50'] = df['close'].rolling(window=50).mean()
#     df['dist_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    
#     # Target: 1 if Price > SMA_20 (Uptrend), 0 if Price < SMA_20 (Downtrend)
#     # We are teaching the AI to recognize "Trends", not predict the future.
#     df['target'] = (df['close'] > sma20).astype(int)
    
#     return df.dropna()

# FEATURES = ['rsi', 'macd', 'macd_signal', 'bb_width', 'dist_sma50']

# # ============================================================
# # 2. SIMPLE DATA LOADER
# # ============================================================
# def load_data(csv_path, scaler_path):
#     df = pd.read_csv(csv_path)
#     df = add_features(df)
    
#     X = df[FEATURES].values
#     y = df['target'].values
    
#     # Scale
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     joblib.dump({'scaler': scaler, 'features': FEATURES}, scaler_path)
#     return X_scaled, y
 
# # ============================================================
# # 3. SIMPLE MODEL (Dense Network)
# # ============================================================
# def build_model(input_dim):
#     model = Sequential([
#         Input(shape=(input_dim,)),
#         Dense(64, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.2),
#         Dense(32, activation='relu'),
#         Dropout(0.2),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
#                   loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # ============================================================
# # 4. TRAINING
# # ============================================================
# def train(csv_file, model_name, scaler_name):
#     print(f"\n🚀 Training {model_name}...")
    
#     X, y = load_data(csv_file, scaler_name)
    
#     # Split
#     split = int(len(X) * 0.8)
#     X_train, y_train = X[:split], y[:split]
#     X_val, y_val = X[split:], y[split:]
    
#     model = build_model(X_train.shape[1])
    
#     model.fit(X_train, y_train, validation_data=(X_val, y_val),
#               epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
#     model.save(model_name)
#     loss, acc = model.evaluate(X_val, y_val, verbose=0)
#     print(f"🏆 Accuracy: {acc:.2%}")

# if __name__ == "__main__":
#     train("data/training/data_5m.csv", "models/model_5m.keras", "models/scaler_5m.pkl")
#     train("data/training/data_15m.csv", "models/model_15m.keras", "models/scaler_15m.pkl")







# Demo 
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# ============================================================
# CONFIGURATION
# ============================================================
LOOKBACK_WINDOW = 60

# Features the model will use to learn patterns
FEATURE_COLUMNS = [
    'close', 'rsi', 'macd', 'macd_signal', 'williams_r', 'momentum', 'ema_20'
]

# Hyperparameters
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.3
RECURRENT_DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 60
L2_REG = 0.001

# ============================================================
# TARGET LABELING STRATEGIES
# ============================================================

def create_target_threshold(df, threshold_pct=0.15, lookahead=5):
    """
    Strategy 1: Threshold-based labeling
    - Predicts if price will move UP by at least `threshold_pct`% in next `lookahead` candles
    - Ignores small/noisy moves (neither strong up nor down)
    
    Parameters:
    - threshold_pct: Minimum percentage move to consider (e.g., 0.15 = 0.15%)
    - lookahead: Number of candles to look ahead
    """
    print(f"📊 Creating target: Threshold={threshold_pct}%, Lookahead={lookahead} candles")
    
    # Calculate future return over `lookahead` candles
    df['future_close'] = df['close'].shift(-lookahead)
    df['future_return'] = (df['future_close'] - df['close']) / df['close'] * 100  # in percentage
    
    # Label: 1 if return > threshold (BUY signal), 0 otherwise
    df['target'] = (df['future_return'] > threshold_pct).astype(int)
    
    # Drop rows where we can't calculate future return
    df.dropna(subset=['target', 'future_return'], inplace=True)
    
    # Clean up temporary columns
    df.drop(columns=['future_close', 'future_return'], inplace=True)
    
    return df


def create_target_triple_barrier(df, profit_target=0.3, stop_loss=0.2, max_holding=10):
    """
    Strategy 2: Triple Barrier Method (Professional Quant Approach)
    - Sets profit target (take profit), stop loss, and max holding period
    - Labels based on which barrier is hit first
    
    Parameters:
    - profit_target: Take profit percentage (e.g., 0.3 = 0.3%)
    - stop_loss: Stop loss percentage (e.g., 0.2 = 0.2%)
    - max_holding: Maximum candles to hold before timeout
    """
    print(f"📊 Creating target: Triple Barrier (TP={profit_target}%, SL={stop_loss}%, MaxHold={max_holding})")
    
    targets = []
    closes = df['close'].values
    
    for i in range(len(df)):
        if i + max_holding >= len(df):
            targets.append(np.nan)
            continue
            
        entry_price = closes[i]
        tp_price = entry_price * (1 + profit_target / 100)
        sl_price = entry_price * (1 - stop_loss / 100)
        
        label = 0  # Default: no clear signal (or stop loss hit)
        
        for j in range(1, max_holding + 1):
            future_price = closes[i + j]
            
            # Check if take profit hit first
            if future_price >= tp_price:
                label = 1  # WIN - BUY signal was correct
                break
            # Check if stop loss hit
            elif future_price <= sl_price:
                label = 0  # LOSS - should not have bought
                break
        
        targets.append(label)
    
    df['target'] = targets
    df.dropna(subset=['target'], inplace=True)
    df['target'] = df['target'].astype(int)
    
    return df


def create_target_trend_following(df, ema_short=5, ema_long=20):
    """
    Strategy 3: Trend Following
    - Labels based on whether short EMA crosses above long EMA (uptrend starting)
    - More suitable for trend-following strategies
    
    Parameters:
    - ema_short: Short EMA period
    - ema_long: Long EMA period
    """
    print(f"📊 Creating target: Trend Following (EMA{ema_short} vs EMA{ema_long})")
    
    df['ema_short'] = df['close'].ewm(span=ema_short, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=ema_long, adjust=False).mean()
    
    # Shift to make it predictive (we want to predict FUTURE crossover)
    df['future_ema_short'] = df['ema_short'].shift(-5)
    df['future_ema_long'] = df['ema_long'].shift(-5)
    
    # Label: 1 if short EMA will be above long EMA (uptrend)
    df['target'] = (df['future_ema_short'] > df['future_ema_long']).astype(int)
    
    # Clean up
    df.drop(columns=['ema_short', 'ema_long', 'future_ema_short', 'future_ema_long'], inplace=True)
    df.dropna(subset=['target'], inplace=True)
    
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_extra_features(df):
    """
    Add more meaningful features that may help the model learn patterns.
    """
    print("🔧 Adding extra technical features...")
    
    # Price changes
    df['price_change_1'] = df['close'].pct_change(1) * 100
    df['price_change_5'] = df['close'].pct_change(5) * 100
    df['price_change_10'] = df['close'].pct_change(10) * 100
    
    # Volatility (rolling std)
    df['volatility_10'] = df['close'].rolling(window=10).std()
    df['volatility_20'] = df['close'].rolling(window=20).std()
    
    # RSI momentum (RSI change)
    if 'rsi' in df.columns:
        df['rsi_change'] = df['rsi'].diff(3)
    
    # MACD histogram
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Price relative to EMA
    if 'ema_20' in df.columns:
        df['price_vs_ema'] = (df['close'] - df['ema_20']) / df['ema_20'] * 100
    
    # High-Low range (if available)
    if 'high' in df.columns and 'low' in df.columns:
        df['hl_range'] = (df['high'] - df['low']) / df['close'] * 100
    
    # Drop NaN rows created by rolling calculations
    df.dropna(inplace=True)
    
    return df


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def load_and_prepare_data(csv_path, scaler_save_path, target_strategy='threshold'):
    """
    Load data, create target, add features, and scale.
    
    Parameters:
    - csv_path: Path to CSV file
    - scaler_save_path: Where to save the scaler
    - target_strategy: 'threshold', 'triple_barrier', or 'trend'
    """
    print(f"\n⏳ Loading dataset from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None, None

    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} rows")
    
    # ---- STEP 1: Add Extra Features ----
    df = add_extra_features(df)
    
    # ---- STEP 2: Create Target Variable ----
    if target_strategy == 'threshold':
        # Good for 5m/15m data - predicts 0.15%+ move in next 5 candles
        df = create_target_threshold(df, threshold_pct=0.15, lookahead=5)
    elif target_strategy == 'triple_barrier':
        # Professional approach - simulates actual trading
        df = create_target_triple_barrier(df, profit_target=0.3, stop_loss=0.2, max_holding=10)
    elif target_strategy == 'trend':
        # Trend following - predicts trend direction
        df = create_target_trend_following(df, ema_short=5, ema_long=20)
    else:
        print(f"❌ Unknown target strategy: {target_strategy}")
        return None, None
    
    # ---- STEP 3: Define Feature Columns ----
    # Base features + new features we added
    feature_cols = FEATURE_COLUMNS.copy()
    
    # Add extra features if they exist
    extra_features = [
        'price_change_1', 'price_change_5', 'price_change_10',
        'volatility_10', 'volatility_20', 'rsi_change',
        'macd_histogram', 'price_vs_ema', 'hl_range'
    ]
    for feat in extra_features:
        if feat in df.columns:
            feature_cols.append(feat)
    
    print(f"📊 Using {len(feature_cols)} features: {feature_cols}")
    
    # Check all required columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing columns (will skip): {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    # ---- STEP 4: Final Cleanup ----
    df.dropna(inplace=True)
    print(f"📊 After cleanup: {len(df)} rows")
    
    if len(df) < LOOKBACK_WINDOW + 100:
        print(f"❌ Not enough data. Need at least {LOOKBACK_WINDOW + 100} rows.")
        return None, None
    
    # ---- STEP 5: Check Class Balance ----
    class_counts = df['target'].value_counts()
    print(f"📊 Class distribution:\n{class_counts}")
    print(f"📊 Class ratio: {class_counts.min() / class_counts.max():.2%}")
    
    # Warn if severely imbalanced
    if class_counts.min() / class_counts.max() < 0.3:
        print("⚠️ Warning: Classes are imbalanced. Consider adjusting threshold.")
    
    # ---- STEP 6: Scale Features ----
    data_x = df[feature_cols].values
    data_y = df['target'].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_x_scaled = scaler.fit_transform(data_x)
    
    # Save scaler and feature columns
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump({
        'scaler': scaler,
        'feature_columns': feature_cols
    }, scaler_save_path)
    print(f"✅ Scaler saved to {scaler_save_path}")
    
    return data_x_scaled, data_y


# ============================================================
# SEQUENCE CREATION
# ============================================================

def create_sequences(data, target, lookback):
    """Create time sequences for LSTM."""
    X, y = [], []
    total = len(data) - lookback
    
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(target[i])
        
        if (i - lookback) % 10000 == 0 and i > lookback:
            print(f"   Processed {i - lookback}/{total} sequences...")

    return np.array(X), np.array(y)


# ============================================================
# MODEL BUILDING
# ============================================================

def build_optimized_model(input_shape):
    """Build optimized LSTM model."""
    model = Sequential([
        Input(shape=input_shape),
        
        Bidirectional(LSTM(
            units=LSTM_UNITS_1,
            return_sequences=True,
            kernel_regularizer=l2(L2_REG),
            recurrent_dropout=RECURRENT_DROPOUT
        )),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        LSTM(
            units=LSTM_UNITS_2,
            return_sequences=False,
            kernel_regularizer=l2(L2_REG),
            recurrent_dropout=RECURRENT_DROPOUT
        ),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        Dense(units=DENSE_UNITS, activation='relu', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE / 2),

        Dense(units=1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model


def get_callbacks(model_path):
    """Get training callbacks."""
    return [
        EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(csv_file, model_name, scaler_name, target_strategy='threshold'):
    """Train model for a specific timeframe."""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING TRAINING: {model_name}")
    print(f"   Target Strategy: {target_strategy}")
    print(f"{'='*60}")

    # 1. Load and Prepare Data
    X_data, y_data = load_and_prepare_data(csv_file, scaler_name, target_strategy)
    if X_data is None:
        return None

    # 2. Create Sequences
    print("\n✂️ Creating Sequences...")
    X, y = create_sequences(X_data, y_data, LOOKBACK_WINDOW)
    print(f"📊 Total sequences: {len(X)}")

    # 3. Train/Test Split (time-series split, no shuffle!)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"📊 Training: {len(X_train)} | Validation: {len(X_test)}")

    # 4. Compute Class Weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"⚖️ Class weights: {class_weight_dict}")

    # 5. Build Model
    model = build_optimized_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    # 6. Train
    print(f"\n🧠 Training (max {EPOCHS} epochs, early stopping on val_auc)...")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=get_callbacks(model_name),
        verbose=1
    )

    # 7. Evaluate
    print(f"\n📈 Final Evaluation:")
    results = model.evaluate(X_test, y_test, verbose=0)
    for name, value in zip(model.metrics_names, results):
        print(f"   {name}: {value:.4f}")

    # 8. Save History
    history_path = model_name.replace('.keras', '_history.csv')
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    print(f"📜 History saved to: {history_path}")

    print(f"\n🎉 TRAINING COMPLETE: {model_name}")
    
    return model, history


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("🔧 TensorFlow version:", tf.__version__)
    print("🔧 GPU Available:", tf.config.list_physical_devices('GPU'))
    
    # ============================================================
    # CHOOSE YOUR TARGET STRATEGY HERE
    # ============================================================
    # Options:
    #   'threshold'      - Predicts if price moves up by X% (recommended to start)
    #   'triple_barrier' - Professional quant approach with TP/SL
    #   'trend'          - Predicts trend direction
    # ============================================================
    
    TARGET_STRATEGY = 'threshold'  # <-- Change this to try different strategies
    
    # --- Train 5 Minute Model ---
    train_model(
        csv_file="data/training/data_5m.csv",
        model_name="models/model_5m.keras",
        scaler_name="models/scaler_5m.pkl",
        target_strategy=TARGET_STRATEGY
    )

    # --- Train 15 Minute Model ---
    train_model(
        csv_file="data/training/data_15m.csv",
        model_name="models/model_15m.keras",
        scaler_name="models/scaler_15m.pkl",
        target_strategy=TARGET_STRATEGY
    )

    print("\n" + "="*60)
    print("✅ ALL MODELS TRAINED!")
    print("="*60)


if __name__ == "__main__":
    main()
