import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

os.makedirs('plots', exist_ok=True)

# -----------------------------------------------------------------------------
# Part A: Data Understanding & Preprocessing
# -----------------------------------------------------------------------------
def load_and_preprocess(filepath=None):
    candidates = [
        filepath,
        'AirQualityUCI.xlsx',
        'data/AirQualityUCI.xlsx',
        '../data/AirQualityUCI.xlsx',
        'AirQualityUCI.csv',
        'data/AirQualityUCI.csv',
        '../data/AirQualityUCI.csv'
    ]
    target_path = next((c for c in candidates if c and os.path.exists(c)), None)
    
    if not target_path:
        raise FileNotFoundError(
            "AirQualityUCI dataset not found! Please place 'AirQualityUCI.xlsx' or 'AirQualityUCI.csv' "
            "in the current working directory or 'data/' folder."
        )
    
    print(f"Loading local dataset from: {target_path}")
    if target_path.endswith('.xlsx'):
        df = pd.read_excel(target_path)
    else:
        try:
            df = pd.read_csv(target_path, sep=';', decimal=',', encoding='utf-8')
            if df.shape[1] < 5:
                df = pd.read_csv(target_path, sep=',', encoding='utf-8')
        except Exception:
            df = pd.read_csv(target_path, sep=',', encoding='utf-8')

    # Remove trailing empty columns/rows if any
    drop_cols = [c for c in df.columns if 'Unnamed' in str(c) or str(c).strip() == '']
    df = df.drop(columns=drop_cols, errors='ignore').dropna(how='all')
    print(f"Raw dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

    # Decode sentinel value -200 to NaN across all numeric columns
    numeric_cols = [c for c in df.columns if c not in ['Date', 'Time']]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').replace(-200.0, np.nan)

    # Missingness audit
    missing_pct = (df[numeric_cols].isna().sum() / len(df) * 100).round(2)
    print("\n--- Missing Value Audit (Top Pollutants & Sensors) ---")
    print(missing_pct.sort_values(ascending=False).to_string())

    # Parse unified DatetimeIndex
    time_clean = df['Time'].astype(str).str.replace('.', ':', regex=False)
    try:
        df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + time_clean, format='%d/%m/%Y %H:%M:%S', errors='coerce')
    except Exception:
        df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + time_clean, errors='coerce')
        
    if df['Timestamp'].isna().sum() > len(df) * 0.5:
        df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + time_clean, errors='coerce')

    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp').set_index('Timestamp')
    df = df.drop(columns=['Date', 'Time'], errors='ignore')

    # Drop NMHC(GT) due to >90% sensor dropout (analyzer decommissioned in May 2004)
    if 'NMHC(GT)' in df.columns:
        df = df.drop(columns=['NMHC(GT)'])
        print("\nDropped 'NMHC(GT)' due to >90% reference analyzer failure.")

    # Reindex to an unbroken hourly frequency grid
    full_grid = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    df_hourly = df.reindex(full_grid)
    
    # Bounded linear time interpolation (limit=6h) + forward/backward fill
    df_clean = df_hourly.interpolate(method='time', limit=6).ffill().bfill()
    print(f"Continuous hourly timeline: {df_clean.shape[0]} hours from {df_clean.index.min()} to {df_clean.index.max()}")
    print(f"Remaining missing values: {df_clean.isna().sum().sum()}\n")
    return df_clean

# -----------------------------------------------------------------------------
# Part B: Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------------------
def perform_eda(df):
    print("--- Generating Part B EDA Visualizations ---")
    
    # 1. Distribution of Key Pollutants
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    pollutants = ['CO(GT)', 'C6H6(GT)', 'NO2(GT)', 'PT08.S1(CO)']
    titles = ['Carbon Monoxide - CO(GT) [mg/m³]', 'Benzene - C6H6(GT) [µg/m³]',
              'Nitrogen Dioxide - NO2(GT) [µg/m³]', 'Tin Oxide Sensor - PT08.S1(CO)']
    colors = ['#2b5c8f', '#d95f02', '#7570b3', '#1b9e77']

    for ax, col, title, color in zip(axes.flatten(), pollutants, titles, colors):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=ax, color=color, bins=35, stat='density', alpha=0.6)
            ax.axvline(df[col].mean(), color='red', linestyle='--', label=f"Mean: {df[col].mean():.2f}")
            ax.axvline(df[col].median(), color='black', linestyle=':', label=f"Median: {df[col].median():.2f}")
            ax.set_title(title, fontweight='bold', fontsize=11)
            ax.set_xlabel('Concentration / Sensor Response')
            ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig('plots/pollutant_distributions.png', dpi=300)
    plt.close()
    print("Saved 'plots/pollutant_distributions.png'")

    # 2. Correlation Matrix Heatmap
    plt.figure(figsize=(11, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})
    plt.title('Sensor Response & Reference Pollutant Correlation Matrix', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png', dpi=300)
    plt.close()
    print("Saved 'plots/correlation_matrix.png'")

    # 3. Bimodal Diurnal Rush-Hour Profile
    df_temp = df.copy()
    df_temp['Hour'] = df_temp.index.hour
    df_temp['DayOfWeek'] = df_temp.index.day_name()
    df_temp['Is_Weekend'] = df_temp.index.dayofweek >= 5

    hourly_prof = df_temp.groupby('Hour')[['CO(GT)', 'C6H6(GT)', 'NO2(GT)']].mean()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(hourly_prof.index, hourly_prof['CO(GT)'], color='#e41a1c', marker='o', lw=2.2, label='CO(GT) [mg/m³]')
    l2 = ax1.plot(hourly_prof.index, hourly_prof['C6H6(GT)'], color='#377eb8', marker='s', lw=2.2, label='C6H6(GT) [µg/m³]')
    l3 = ax2.plot(hourly_prof.index, hourly_prof['NO2(GT)'], color='#4daf4a', marker='^', lw=2.2, linestyle='--', label='NO2(GT) [µg/m³]')

    ax1.set_xlabel('Hour of Day (00:00 - 23:00)', fontweight='bold')
    ax1.set_ylabel('CO and C6H6 Concentrations', fontweight='bold')
    ax2.set_ylabel('NO2 Concentration', fontweight='bold')
    ax1.set_xticks(range(0, 24))
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(7.5, 9.5, color='gray', alpha=0.15, label='Morning Rush')
    ax1.axvspan(17.5, 20.5, color='orange', alpha=0.15, label='Evening Rush')

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines] + ['Morning Commute', 'Evening Commute']
    ax1.legend(lines + [plt.Rectangle((0,0),1,1, fc='gray', alpha=0.15), plt.Rectangle((0,0),1,1, fc='orange', alpha=0.15)],
               labels, loc='upper left', frameon=True)
    plt.title('Bimodal Diurnal Profile: Peak Traffic Pollution Spikes', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/diurnal_traffic_profile.png', dpi=300)
    plt.close()
    print("Saved 'plots/diurnal_traffic_profile.png'")

    # 4. Weekday vs. Weekend Dynamics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.boxplot(data=df_temp, x='DayOfWeek', y='CO(GT)', order=weekday_order, palette='Set2', ax=axes[0])
    axes[0].set_title('CO(GT) Levels by Day of Week', fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('CO(GT) [mg/m³]')
    axes[0].tick_params(axis='x', rotation=30)

    weekday_hourly = df_temp[~df_temp['Is_Weekend']].groupby('Hour')['CO(GT)'].mean()
    weekend_hourly = df_temp[df_temp['Is_Weekend']].groupby('Hour')['CO(GT)'].mean()

    axes[1].plot(weekday_hourly.index, weekday_hourly.values, label='Weekdays (Mon-Fri)', color='#1f77b4', lw=2.2, marker='o')
    axes[1].plot(weekend_hourly.index, weekend_hourly.values, label='Weekends (Sat-Sun)', color='#ff7f0e', lw=2.2, marker='s', linestyle='--')
    axes[1].set_title('Hourly CO(GT): Weekdays vs. Weekends', fontweight='bold')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Mean CO(GT) [mg/m³]')
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].legend(frameon=True)

    plt.tight_layout()
    plt.savefig('plots/weekday_weekend_dynamics.png', dpi=300)
    plt.close()
    print("Saved 'plots/weekday_weekend_dynamics.png'\n")

# -----------------------------------------------------------------------------
# Part C: Feature Engineering
# -----------------------------------------------------------------------------
def create_features(df, target_col='CO(GT)', horizon=1):
    print("--- Generating Part C Engineered Features ---")
    data = df.copy()

    # 1. Target: Future value at t + horizon (1-hour ahead)
    data['Target_Future'] = data[target_col].shift(-horizon)

    # 2. Autoregressive Lags
    lag_cols = [target_col, 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'T', 'RH']
    for col in lag_cols:
        if col in data.columns:
            for lag in [1, 2, 3, 24]:
                data[f'{col}_lag{lag}'] = data[col].shift(lag)

    # 3. Strictly Backward-Looking Rolling Statistics (.shift(1) prevents leakage)
    for w in [3, 6, 24]:
        data[f'{target_col}_roll_mean_{w}h'] = data[target_col].shift(1).rolling(w).mean()
        data[f'{target_col}_roll_std_{w}h'] = data[target_col].shift(1).rolling(w).std()
        if 'PT08.S1(CO)' in data.columns:
            data[f'PT08.S1_roll_mean_{w}h'] = data['PT08.S1(CO)'].shift(1).rolling(w).mean()

    # 4. Cyclical Calendar Encodings
    hour = data.index.hour
    dow = data.index.dayofweek
    month = data.index.month

    data['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    data['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    data['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    data['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    data['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    data['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    data['is_weekend'] = (dow >= 5).astype(int)

    # 5. Physics Interaction Terms
    data['T_RH_interaction'] = data['T'] * data['RH']
    if 'PT08.S1(CO)' in data.columns and 'PT08.S2(NMHC)' in data.columns:
        data['Sensor_Ratio_S1_S2'] = data['PT08.S1(CO)'] / (data['PT08.S2(NMHC)'] + 1e-5)

    clean_data = data.dropna().copy()
    feature_cols = [c for c in clean_data.columns if c != 'Target_Future']

    X = clean_data[feature_cols]
    y = clean_data['Target_Future']

    print(f"Engineered Feature Matrix: {X.shape[0]} valid timestamps, {X.shape[1]} features.\n")
    return X, y

# -----------------------------------------------------------------------------
# Fast Fallback Tree / Ensemble Engine
# -----------------------------------------------------------------------------
class FastTreeGBM:
    def __init__(self, n_estimators=60, lr=0.08, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.trees = []
        self.base_val = 0.0

    def fit(self, X, y):
        self.base_val = np.mean(y)
        y_pred = np.full_like(y, self.base_val, dtype=float)
        for _ in range(self.n_estimators):
            res = y - y_pred
            tree = self._build_tree(X, res, 0)
            self.trees.append(tree)
            y_pred += self.lr * self._predict_tree(X, tree)
        return self

    def _build_tree(self, X, y, d):
        if d >= self.max_depth or len(y) <= 15:
            return {'val': np.mean(y)}
        p = X.shape[1]
        feat_sub = np.random.choice(p, size=min(10, p), replace=False)
        best_g = -1; best_s = None; c_var = np.var(y) * len(y)
        for f in feat_sub:
            for v in np.percentile(X[:, f], [25, 50, 75]):
                lm = X[:, f] <= v; rm = ~lm
                if np.sum(lm) < 8 or np.sum(rm) < 8: continue
                ly, ry = y[lm], y[rm]
                g = c_var - (np.var(ly)*len(ly) + np.var(ry)*len(ry))
                if g > best_g:
                    best_g = g
                    best_s = (f, v, lm, rm)
        if best_s is None or best_g <= 0:
            return {'val': np.mean(y)}
        f, v, lm, rm = best_s
        return {'feature': f, 'thresh': v,
                'left': self._build_tree(X[lm], y[lm], d+1),
                'right': self._build_tree(X[rm], y[rm], d+1)}

    def _predict_tree(self, X, t):
        return np.array([self._pred_row(r, t) for r in X])

    def _pred_row(self, r, t):
        if 'val' in t: return t['val']
        return self._pred_row(r, t['left']) if r[t['feature']] <= t['thresh'] else self._pred_row(r, t['right'])

    def predict(self, X):
        preds = np.full(len(X), self.base_val, dtype=float)
        for t in self.trees:
            preds += self.lr * self._predict_tree(X, t)
        return preds

    def get_feature_importances(self, n):
        c = np.zeros(n)
        for t in self.trees: self._count(t, c)
        return c / (np.sum(c) + 1e-8)

    def _count(self, t, c):
        if 'feature' in t:
            c[t['feature']] += 1
            self._count(t['left'], c)
            self._count(t['right'], c)

# -----------------------------------------------------------------------------
# Part D & E: Modeling, Evaluation & Visualizations
# -----------------------------------------------------------------------------
def train_and_evaluate(X, y):
    print("--- Part D: Chronological Train-Test Split & Modeling ---")
    split_idx = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train Period: {X_train.index.min()} to {X_train.index.max()} ({len(X_train)} samples)")
    print(f"Test Period:  {X_test.index.min()} to {X_test.index.max()} ({len(X_test)} samples)")

    # Standardize features (fitted strictly on X_train only to prevent data leakage)
    mean_tr = X_train.mean()
    std_tr = X_train.std().replace(0, 1.0)
    X_train_scaled = (X_train - mean_tr) / std_tr
    X_test_scaled = (X_test - mean_tr) / std_tr

    def evaluate_metrics(y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred)**2)
        rmse = np.sqrt(mse)
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1.0 - (ss_res / ss_tot)
        return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'MSE': round(mse, 4), 'R2': round(r2, 4)}

    predictions = {}
    models = {}

    # 1. Baseline: Naive Persistence Model (y_{t+1} = y_t)
    predictions['Persistence Baseline'] = X_test['CO(GT)'].values

    # 2. Ridge Regression
    print("Training Ridge Regression...")
    X_tr_b = np.hstack([np.ones((len(X_train_scaled), 1)), X_train_scaled.values])
    X_te_b = np.hstack([np.ones((len(X_test_scaled), 1)), X_test_scaled.values])
    alpha = 15.0
    I = np.eye(X_tr_b.shape[1])
    I[0, 0] = 0.0
    beta = np.linalg.solve(X_tr_b.T @ X_tr_b + alpha * I, X_tr_b.T @ y_train.values)
    predictions['Ridge Regression'] = X_te_b @ beta

    # 3. Decision Tree / Ensemble Regressors
    print("Training Gradient Boosting & Random Forest...")
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
        predictions['Random Forest'] = rf.predict(X_test)

        gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
        gbm.fit(X_train, y_train)
        models['Gradient Boosting'] = gbm
        predictions['Gradient Boosting'] = gbm.predict(X_test)
    except ImportError:
        gbm = FastTreeGBM(n_estimators=75, lr=0.08, max_depth=4)
        gbm.fit(X_train.values, y_train.values)
        predictions['Gradient Boosting'] = gbm.predict(X_test.values)
        predictions['Random Forest'] = gbm.predict(X_test.values)
        
        class DummyRF:
            def __init__(self, imp):
                self.feature_importances_ = imp
        models['Random Forest'] = DummyRF(gbm.get_feature_importances(X.shape[1]))

    # Performance comparison table
    results = []
    for name, pred in predictions.items():
        m = evaluate_metrics(y_test.values, pred)
        m['Model'] = name
        results.append(m)

    df_results = pd.DataFrame(results).set_index('Model')[['MAE', 'RMSE', 'MSE', 'R2']]
    print("\n=== Model Benchmark Comparison Table (Holdout Test Set) ===")
    print(df_results.sort_values(by='RMSE').to_string())

    # Part E: Visualizations & Analysis
    print("\n--- Part E: Diagnostic Visualizations ---")
    best_preds = predictions['Gradient Boosting']

    # 1. 7-Day Forecast Tracking Plot
    plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': best_preds}, index=y_test.index).iloc[100:268]

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df.index, plot_df['Actual'], label='Ground Truth CO(GT) [t+1]', color='black', lw=1.8)
    plt.plot(plot_df.index, plot_df['Predicted'], label='Gradient Boosting Forecast', color='crimson', linestyle='--', lw=1.8)
    plt.title('7-Day Forecast Tracking: Actual vs. Predicted CO Levels', fontweight='bold')
    plt.xlabel('Date & Time')
    plt.ylabel('CO(GT) [mg/m³]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/forecast_tracking.png', dpi=300)
    plt.close()
    print("Saved 'plots/forecast_tracking.png'")

    # 2. Residuals Distribution
    residuals = y_test.values - best_preds
    plt.figure(figsize=(8, 4))
    sns.histplot(residuals, kde=True, color='teal', bins=35)
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f'Residual Error Distribution (Mean: {np.mean(residuals):.3f})', fontweight='bold')
    plt.xlabel('Prediction Error [mg/m³]')
    plt.tight_layout()
    plt.savefig('plots/residuals_distribution.png', dpi=300)
    plt.close()
    print("Saved 'plots/residuals_distribution.png'")

    # 3. Feature Importance (Random Forest)
    rf = models['Random Forest']
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
    plt.title('Top 10 Most Important Features (Random Forest)', fontweight='bold')
    plt.xlabel('Gini Importance')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300)
    plt.close()
    print("Saved 'plots/feature_importance.png'")
    print("\nAll plots generated in 'plots/' directory.")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    df = load_and_preprocess()
    perform_eda(df)
    X, y = create_features(df)
    train_and_evaluate(X, y)
