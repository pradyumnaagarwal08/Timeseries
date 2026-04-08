import numpy as np
import pandas as pd


def forecast_future(model, scaler, df, seq_length=24, days=30):

    # Get column name (IMPORTANT)
    energy_col = df.columns[0]

    # Get last sequence
    last_sequence = df[energy_col].values[-seq_length:]

    # Reshape for scaler
    current_sequence = last_sequence.reshape(-1, 1)

    future_predictions = []

    for _ in range(days * 24):

        # Scale input
        scaled_input = scaler.transform(current_sequence)

        # Reshape for LSTM
        scaled_input = scaled_input.reshape(1, seq_length, 1)

        # Predict
        pred_scaled = model.predict(scaled_input, verbose=0)

        # Inverse scale
        pred = scaler.inverse_transform(pred_scaled)[0][0]

        future_predictions.append(pred)

        # Update sequence (VERY IMPORTANT)
        current_sequence = np.vstack([
            current_sequence[1:], 
            [[pred]]
        ])

    # Create future timestamps
    future_index = pd.date_range(
        start=df.index[-1] + pd.Timedelta(hours=1),
        periods=days * 24,
        freq="H"
    )

    # ✅ FIXED COLUMN NAME
    forecast_df = pd.DataFrame(
        future_predictions,
        index=future_index,
        columns=[energy_col]   # 🔥 THIS FIXES YOUR ERROR
    )

    return forecast_df