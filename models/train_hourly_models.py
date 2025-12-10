import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization,
    Bidirectional, GaussianNoise, Flatten, RepeatVector,
    Permute, Multiply, Lambda, Activation
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l1_l2

# ============================================================
# GLOBAL CONFIG
# ============================================================
DATA_PATH = "data/training/full_history_data.csv"  # 1H data file

# Symbols to train (you can adjust this list)
SYMBOLS_TO_TRAIN = [
    "BTC-USD",
    "ETH-USD",
    "XRP-USD",
    "EURUSD=X",
    "GBPUSD=X",
    "NVDA",
    "AAPL"
]

# Sequence length (how many past hours to look at)
LOOKBACK_WINDOW = 48   # 2 days of hourly candles

# Core features from your CSV
BASE_FEATURES = [
    "close", "open", "high", "low", "volume",
    "rsi", "macd", "macd_signal", "williams_r",
    "momentum", "sma_20", "ema_20", "awesome_osc"
]

# Model size
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 32

# Regularization
DROPOUT_RATE = 0.4
RECURRENT_DROPOUT = 0.2
L1_REG = 0.0005
L2_REG = 0.005
NOISE_STDDEV = 0.01

# Training
LEARNING_RATE = 0.0005
BATCH_SIZE = 256
EPOCHS = 100

# Threshold for trading later
CONFIDENCE_THRESHOLD = 0.6


# ============================================================
# DATA LOADING & PREP
# ============================================================

def load_symbol_data(symbol, path=DATA_PATH):
    if not os.path.exists(path):
        print(f"❌ Data file not found: {path}")
        return None

    print(f"\n⏳ Loading data for {symbol} from {path}")
    df = pd.read_csv(path)

    # Filter symbol
    df = df[df["symbol"] == symbol].copy()
    df.sort_values("Datetime", inplace=True)

    if df.empty:
        print(f"❌ No data for symbol: {symbol}")
        return None

    print(f"📊 Rows for {symbol}: {len(df)}")

    # Ensure required columns exist
    required_cols = set(BASE_FEATURES + ["target", "future_close"])
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ Missing columns for {symbol}: {missing}")
        return None

    # Use the existing 'target' column from your CSV (already labeled)
    # 1 = future_close up enough, 0 = not up
    class_counts = df["target"].value_counts()
    print(f"📊 Class distribution ({symbol}):")
    for cls, cnt in class_counts.items():
        print(f"   Class {cls}: {cnt}")
    if len(class_counts) < 2:
        print("❌ Only one class present, skipping.")
        return None

    # Select features
    feature_cols = BASE_FEATURES.copy()

    # Clean data
    df = df[feature_cols + ["target"]].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"📊 After cleanup ({symbol}): {len(df)} rows")

    if len(df) < LOOKBACK_WINDOW + 200:
        print("❌ Not enough data after cleanup, skipping.")
        return None

    X = df[feature_cols].values
    y = df["target"].values

    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -5, 5)

    # Save scaler
    os.makedirs("models", exist_ok=True)
    scaler_path = f"models/scaler_{symbol.replace('=','').replace('-','')}_1h.pkl"
    joblib.dump({
        "scaler": scaler,
        "feature_columns": feature_cols,
        "lookback": LOOKBACK_WINDOW,
        "symbol": symbol
    }, scaler_path)
    print(f"✅ Scaler saved: {scaler_path}")

    return X_scaled, y


def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ============================================================
# MODEL
# ============================================================

def build_model(input_shape):
    inputs = Input(shape=input_shape)

    x = GaussianNoise(NOISE_STDDEV)(inputs)

    x = Bidirectional(LSTM(
        LSTM_UNITS_1,
        return_sequences=True,
        dropout=DROPOUT_RATE,
        recurrent_dropout=RECURRENT_DROPOUT,
        kernel_regularizer=l1_l2(L1_REG, L2_REG),
        recurrent_regularizer=l1_l2(L1_REG / 10, L2_REG / 10)
    ))(x)
    x = BatchNormalization()(x)

    x = LSTM(
        LSTM_UNITS_2,
        return_sequences=True,
        dropout=DROPOUT_RATE,
        recurrent_dropout=RECURRENT_DROPOUT,
        kernel_regularizer=l1_l2(L1_REG, L2_REG)
    )(x)
    x = BatchNormalization()(x)

    # Simple attention over time dimension
    attn = Dense(1, activation="tanh")(x)
    attn = Flatten()(attn)
    attn = Activation("softmax")(attn)
    attn = RepeatVector(LSTM_UNITS_2)(attn)
    attn = Permute((2, 1))(attn)

    x = Multiply()([x, attn])
    x = Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

    x = Dense(DENSE_UNITS, activation="relu", kernel_regularizer=l1_l2(L1_REG, L2_REG))(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


def get_callbacks(model_path):
    return [
        EarlyStopping(
            monitor="val_auc",
            patience=15,
            mode="max",
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1
        )
    ]


# ============================================================
# TRAINING PER SYMBOL
# ============================================================

def train_for_symbol(symbol):
    print("\n" + "=" * 60)
    print(f"🚀 TRAINING 1H MODEL FOR: {symbol}")
    print("=" * 60)

    data = load_symbol_data(symbol)
    if data is None:
        return

    X_scaled, y = data

    # Sequences
    X_seq, y_seq = create_sequences(X_scaled, y, LOOKBACK_WINDOW)
    print(f"📊 Total sequences: {len(X_seq)}")

    # Split: 70% train, 15% val, 15% test
    n = len(X_seq)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]

    print(f"📊 Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"⚖️ Class weights: {class_weight_dict}")

    # Build model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    model_path = f"models/model_{symbol.replace('=','').replace('-','')}_1h.keras"
    callbacks = get_callbacks(model_path)

    print(f"\n🧠 Training (max {EPOCHS} epochs)...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # =====================================================
    # EVALUATION
    # =====================================================
    print("\n" + "=" * 60)
    print(f"📈 EVALUATION FOR {symbol} (TEST SET)")
    print("=" * 60)

    results = model.evaluate(X_test, y_test, verbose=0)
    for name, val in zip(model.metrics_names, results):
        print(f"   {name}: {val:.4f}")

    y_proba = model.predict(X_test, verbose=0).flatten()

    # Default threshold 0.5
    y_pred_05 = (y_proba > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_05).ravel()
    print("\n📊 Confusion Matrix (threshold=0.5):")
    print(f"   TN: {tn}  FP: {fp}")
    print(f"   FN: {fn}  TP: {tp}")
    prec_05 = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_05 = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"   Precision: {prec_05:.2%}")
    print(f"   Recall:    {rec_05:.2%}")

    # High-confidence trades
    print(f"\n📊 High-Confidence (p > {CONFIDENCE_THRESHOLD}):")
    mask = y_proba > CONFIDENCE_THRESHOLD
    total_signals = mask.sum()
    if total_signals > 0:
        correct = ((y_pred_05 == y_test) & mask).sum()
        win_rate = correct / total_signals
        print(f"   Signals: {total_signals}")
        print(f"   Correct: {correct}")
        print(f"   Win Rate: {win_rate:.2%}")
    else:
        print("   No signals above threshold")

    # Save history
    hist_path = model_path.replace(".keras", "_history.csv")
    pd.DataFrame(history.history).to_csv(hist_path, index=False)
    print(f"\n📜 History saved: {hist_path}")
    print(f"🎉 DONE: {symbol}")


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    print("=" * 60)
    print("🔧 HOURLY (1H) LSTM TRAINING - SYMBOL-WISE")
    print("=" * 60)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")

    for sym in SYMBOLS_TO_TRAIN:
        train_for_symbol(sym)

    print("\n" + "=" * 60)
    print("✅ ALL HOURLY MODELS TRAINED")
    print("=" * 60)


if __name__ == "__main__":
    main()
