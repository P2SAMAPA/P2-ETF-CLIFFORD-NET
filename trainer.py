import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from clifford_net import CliffordNetWrapper

def build_multivector_features(returns_df, macro_df, corr_window=60):
    """Same as before: returns 11-dim features per ETF."""
    n_etfs = returns_df.shape[1]
    etf_names = returns_df.columns.tolist()
    # Grade 0: latest return
    last_returns = returns_df.iloc[-1].values.reshape(-1, 1)
    # Grade 1: macro sensitivities
    macro_returns = macro_df.diff().dropna()
    recent_returns = returns_df.iloc[-macro_returns.shape[0]:] if len(returns_df) > len(macro_returns) else macro_returns
    common = recent_returns.index.intersection(macro_returns.index)
    if len(common) < 20:
        macro_sens = np.zeros((n_etfs, len(config.MACRO_COLUMNS)))
    else:
        ret_aligned = recent_returns.loc[common]
        macro_aligned = macro_returns.loc[common]
        macro_sens = []
        for etf in etf_names:
            sens = [ret_aligned[etf].corr(macro_aligned[mcol]) if not np.isnan(ret_aligned[etf].corr(macro_aligned[mcol])) else 0.0 for mcol in config.MACRO_COLUMNS]
            macro_sens.append(sens)
        macro_sens = np.array(macro_sens)
    # Grade 2: bivector (top 6 correlations)
    recent_corr = returns_df.iloc[-corr_window:].corr().values
    bivectors = []
    for i in range(n_etfs):
        corrs = [recent_corr[i, j] for j in range(n_etfs) if j != i]
        top6 = sorted(corrs, reverse=True)[:6] if len(corrs) >= 6 else corrs + [0.0]*(6-len(corrs))
        bivectors.append(top6)
    bivectors = np.array(bivectors)
    features = np.concatenate([last_returns, macro_sens, bivectors], axis=1)
    return features, etf_names

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Full Clifford Net) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < config.TRAIN_WINDOW + 100:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        macro_df = data_manager.get_macro_data(df)
        if macro_df.empty:
            macro_df = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLUMNS)

        # Build daily multivector features and targets
        daily_features, daily_targets = [], []
        start_idx = max(0, len(returns) - config.TRAIN_WINDOW - 50)
        for i in range(start_idx, len(returns)-1):
            window_returns = returns.iloc[:i+1]
            if len(window_returns) < 60:
                continue
            features, _ = build_multivector_features(window_returns, macro_df, corr_window=60)
            target = returns.iloc[i+1].values
            daily_features.append(features)
            daily_targets.append(target)

        if len(daily_features) < 50:
            print("  Not enough daily samples")
            continue

        X = torch.tensor(np.array(daily_features), dtype=torch.float32)
        y = torch.tensor(np.array(daily_targets), dtype=torch.float32)
        # Flatten across days and ETFs
        X_flat = X.view(-1, X.shape[-1])
        y_flat = y.view(-1)
        valid = ~torch.isnan(y_flat)
        X_flat = X_flat[valid]
        y_flat = y_flat[valid]

        if len(X_flat) < 100:
            continue

        split = int(0.8 * len(X_flat))
        X_train, X_val = X_flat[:split], X_flat[split:]
        y_train, y_val = y_flat[:split], y_flat[split:]

        net = CliffordNetWrapper(input_mv_dim=1, hidden_mv_dim=8, output_mv_dim=1)
        optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.MSELoss()

        print("  Training full Clifford network...")
        for epoch in range(config.EPOCHS):
            net.train()
            idx = torch.randperm(len(X_train))
            for i in range(0, len(idx), config.BATCH_SIZE):
                batch_idx = idx[i:i+config.BATCH_SIZE]
                Xb = X_train[batch_idx]
                yb = y_train[batch_idx]
                pred = net(Xb).squeeze()
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                val_pred = net(X_val).squeeze()
                val_loss = criterion(val_pred, y_val)
            if (epoch+1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{config.EPOCHS}, train loss: {loss.item():.4f}, val loss: {val_loss.item():.4f}")

        # Predict for latest day
        last_features, etf_names = build_multivector_features(returns, macro_df, corr_window=60)
        last_tensor = torch.tensor(last_features, dtype=torch.float32)
        with torch.no_grad():
            pred_returns = net(last_tensor).squeeze().numpy()
        sorted_idx = np.argsort(pred_returns)[::-1]
        top_etfs = []
        full_scores = {}
        for i, idx in enumerate(sorted_idx):
            ticker = etf_names[idx]
            pred = pred_returns[idx]
            full_scores[ticker] = pred
            if i < config.TOP_N:
                top_etfs.append({"ticker": ticker, "pred_return": float(pred)})
        print(f"  Top 3 ETFs: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/clifford_net_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Full Clifford Geometric Algebra Network complete ===")

if __name__ == "__main__":
    main()
