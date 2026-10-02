# Air Quality Forecasting - Technical Assessment Submission

## 2. Tasks Completed
* **Task 1: Air Quality Forecasting** (End-to-End Predictive Pipeline)
  * **Part A: Data Understanding & Preprocessing** (Sentinel value decoding, analyzer dropout audit, continuous hourly time grid reindexing, bounded time-based linear interpolation)
  * **Part B: Exploratory Data Analysis** (Pollutant distribution analysis, sensor cross-correlation matrix, bimodal diurnal rush-hour profiling, weekday vs. weekend dynamics)
  * **Part C: Feature Engineering** (Future target lag formulation, autoregressive lags, strictly shifted backward-looking rolling statistics, cyclical trigonometric temporal encodings, thermodynamic interaction terms)
  * **Part D: Prediction & Modeling** (Chronological 80/20 train-test split, isolated preprocessor scaling, Persistence Baseline, Ridge Regression, Random Forest, Gradient Boosting; evaluated via MAE, MSE, RMSE, R²)
  * **Part E: Analysis & Residual Diagnostics** (10-day actual vs. predicted tracking trace, residual distribution analysis, Gini feature importance ranking, rigorous temporal data leakage audit)
  * **Part F: Key Findings, Limitations & Next Steps** (7 key empirical findings, 3 observational/sensor limitations, concrete future research roadmap)
  * **Bonus Modules:** Multi-step forecasting horizon degradation analysis (1h, 3h, 6h, 12h, 24h) and 3-tier Air Quality Index (AQI) health alert classification

## 3. Problem Statement
Urban air pollution causes significant environmental and public health concerns. While industrial-grade certified analyzers provide accurate measurements of pollutants like Carbon Monoxide (CO), Benzene (C6H6), and Nitrogen Oxides (NOx/NO2), they are expensive to deploy widely. Low-cost chemical multisensors (PT08.S1 to PT08.S5) provide continuous monitoring but can suffer from cross-sensitivity to temperature and humidity.

The goal of this project is to analyze hourly multi-sensor responses from the UCI Air Quality dataset (collected over one year in an Italian city) and build machine learning models to forecast future air quality (specifically 1-hour ahead CO concentration, CO(GT) at t+1, along with multi-hour lead times) while preventing temporal data leakage.

## 4. Approach
The project follows a structured data science workflow:

### Part A: Data Understanding and Preprocessing
1. Data Ingestion: Loaded the dataset handling European delimiter formatting (semicolon separators and decimal commas).
2. Sentinel Value Decoding: Missing values in the dataset are recorded as -200. These were converted to NaN.
3. Handling High Missingness: Inspected missing percentages across all columns. NMHC(GT) had over 90% missing values because the certified analyzer failed early in the measurement campaign; this column was dropped to prevent synthetic bias.
4. Hourly Indexing and Imputation: Combined Date and Time into an unbroken hourly DatetimeIndex spanning March 2004 to April 2005 (9,357 hours). Sporadic missing readings in the remaining variables were imputed using bounded linear time interpolation (limit of 6 hours) followed by forward/backward fill.

### Part B: Exploratory Data Analysis
1. Pollutant Distributions: Histograms and KDE plots showed positive right-skewed log-normal distributions for CO, C6H6, and NOx.
2. Correlation Analysis: Metal oxide sensor PT08.S2 showed strong correlation with Benzene (r = 0.98), and PT08.S1 showed strong correlation with CO (r = 0.88).
3. Diurnal Profiles: Averaged hourly readings revealed a bimodal daily curve with peaks during morning rush hours (08:00 - 09:00) and evening rush hours (18:00 - 20:00).
4. Weekly Variations: Boxplots showed significant differences between weekdays and weekends, with Sunday pollution levels dropping by more than 45%.

### Part C: Feature Engineering
1. Target Variable: Future Carbon Monoxide concentration at t+1 hour.
2. Autoregressive Lags: Created lagged values (t-1, t-2, t-3, and t-24) for the target and primary sensors to capture short-term momentum and daily periodicity.
3. Rolling Statistics: Calculated 3-hour, 6-hour, and 24-hour backward-looking rolling means and standard deviations using shifted series (.shift(1)) to prevent data leakage.
4. Cyclical Encodings: Transformed Hour, Day of Week, and Month into sine and cosine coordinates to maintain continuous circular transitions across midnight and calendar periods.
5. Domain Interactions: Created temperature and relative humidity interaction terms (T * RH) and sensor impedance ratios to capture atmospheric moisture effects.

### Part D: Prediction
1. Chronological Split: Partitioned the data chronologically into 80% training (7,465 hours) and 20% holdout testing (1,867 hours). Random K-fold splitting was avoided to preserve temporal causality.
2. Feature Scaling: Standardized numerical inputs using a scaler fit strictly on the training set.
3. Models Evaluated:
   * Naive Persistence Baseline (predicts value at t for t+1)
   * Ridge Regression (L2-regularized linear model)
   * Random Forest Regressor
   * Gradient Boosting Regressor
4. Metrics: Evaluated using MAE, MSE, RMSE, and R2.

### Part E: Analysis
1. Forecast Tracking: Plotted 10-day continuous tracking traces comparing actual vs. predicted values.
2. Residual Diagnostics: Analyzed residual distributions (mean centered near zero) and scatter plots of predicted vs. actual values.
3. Feature Importance: Ranked features using Gini split importance from tree models.
4. Data Leakage Prevention: Documented precautions taken, including chronological splitting, shifted rolling windows, and isolated preprocessing transformations.

### Part F: Insights
1. Summarized 7 key empirical takeaways, 3 hardware and observational limitations, and future directions.

## 5. Technologies Used
* Python 3.10+
* pandas: Time-series manipulation, hourly reindexing, rolling features
* numpy: Numerical linear algebra, cyclical trigonometric transforms
* scikit-learn: Ridge, RandomForestRegressor, GradientBoostingRegressor, StandardScaler, metrics
* matplotlib & seaborn: Statistical visualizations and residual diagnostic plots
* openpyxl: Reading and writing Excel sensor datasets

## 6. Results

### Model Performance Comparison (Holdout Test Set, 1-Hour Ahead CO Prediction)

| Model | MAE [mg/m3] | RMSE [mg/m3] | MSE [mg/m3] | R2 Score |
| :--- | :---: | :---: | :---: | :---: |
| Random Forest | 0.7104 | 1.0622 | 1.1282 | 0.5807 |
| Gradient Boosting | 0.7131 | 1.0636 | 1.1313 | 0.5795 |
| Ridge Regression | 0.7948 | 1.1739 | 1.3781 | 0.4878 |
| Persistence Baseline | 1.0251 | 1.5451 | 2.3875 | 0.1127 |

The tree-based ensemble models (Random Forest and Gradient Boosting) performed best, explaining approximately 58% of the out-of-sample variance and lowering RMSE to 1.06 mg/m3, compared to 1.55 mg/m3 for the naive persistence baseline. Ridge regression achieved R2 = 0.49, demonstrating that while linear trends capture baseline drift, non-linear tree models better capture sensor saturation at peak levels.

### Multi-Step Ahead Forecasting Horizon Analysis

| Horizon | MAE [mg/m3] | RMSE [mg/m3] | R2 Score |
| :--- | :---: | :---: | :---: |
| 1 Hour Ahead | 0.7298 | 1.1070 | 0.5445 |
| 3 Hours Ahead | 0.8014 | 1.2214 | 0.4457 |
| 6 Hours Ahead | 0.7853 | 1.1939 | 0.4704 |
| 12 Hours Ahead | 0.7866 | 1.1654 | 0.4956 |
| 24 Hours Ahead | 0.7763 | 1.1643 | 0.4969 |

Prediction accuracy drops between 1h and 3h as short-term persistence decreases, but stabilizes around 24h as the daily cyclical pattern (lag-24 and cyclical hour features) provides recurring predictive value.

### AQI Health Alert Classification
Mapping predictions into standard categories (Good <= 2.0, Moderate 2.0 - 4.5, Unhealthy > 4.5 mg/m3) achieved 73.1% overall alert classification accuracy across the test set.

## 7. Key Learnings
1. Avoiding Data Leakage in Time Series: Random K-fold splitting must not be used on time-series data because adjacent hours are strongly autocorrelated. Using a chronological split and shifted rolling windows (.shift(1)) ensures that the model only learns from historical information.
2. Value of Domain Feature Engineering: Incorporating 24-hour seasonal lags, cyclical trigonometric coordinates, and temperature-humidity interaction terms significantly improved model accuracy across both linear and non-linear models.
3. Handling High Missingness Appropriately: Over 90% of NMHC(GT) was missing due to hardware analyzer decommissioning. Dropping the column rather than imputing thousands of synthetic rows prevented artificial noise from biasing the models.
4. Non-Linear Behavior in Chemical Sensors: Metal oxide sensors exhibit saturation at high gas concentrations. Tree-based ensembles outperformed linear models by effectively capturing these non-linear response curves.

## 8. Challenges and Solutions
1. Challenge: NMHC(GT) had 90.2% missing values due to analyzer decommissioning early in the collection campaign.
   Solution: Dropped NMHC(GT) from the modeling dataset rather than fabricating data with imputation. Retained the titania sensor PT08.S2(NMHC), which remained active and provided strong proxy information for hydrocarbons.
2. Challenge: Preventing rolling feature calculations from leaking target values at time t into predictions for t+1.
   Solution: Shifted all input series by 1 step before calculating rolling means and standard deviations, ensuring strictly backward-looking features.
3. Challenge: Handling European CSV formatting with semicolon delimiters and decimal commas.
   Solution: Built a data loader supporting both semicolon/comma formats and Excel files, properly converting numeric fields and sentinel values.

## 9. Setup and Reproduction

### Running Locally
```bash
# Clone the repository
git clone https://github.com/Yash5217/AIML-Recruitment-2026-YashAgarwal.git
cd AIML-Recruitment-2026-YashAgarwal

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the training script
python3 Task-1-Air-Quality-Forecasting/train_task1.py
```

### Running in Google Colab / Jupyter
Open `Task-1-Air-Quality-Forecasting/Air_Quality_Forecasting_Project.ipynb` in Google Colab or Jupyter Notebook and run all cells to reproduce the analysis and plots.
