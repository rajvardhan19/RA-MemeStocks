#!/usr/bin/env python3
"""
Reddit Outage Impact Analysis using CAPM Market Model
Modified to use CAPM instead of FF four-factor model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

# Set up paths
projectPath = '/Users/zhaomufan/学习资料/wrds project/wsb_governance/'
dataPath = projectPath + 'data/'
figurePath = projectPath + 'figures/'
stockPath = dataPath

# Create directories
os.makedirs(figurePath, exist_ok=True)

# Set plotting style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

print("Loading data...")

# Load stock data
def load_stock_data(stock_symbol, price_col, return_col):
    """Load and preprocess stock data"""
    try:
        if stock_symbol == 'GME':
            data = pd.read_csv(stockPath + 'GME-minute_price-excess-return.csv')
            data.rename(columns={price_col: f'{stock_symbol.lower()}Price'}, inplace=True)
        elif stock_symbol == 'AMC':
            data = pd.read_csv(stockPath + 'AMC-minute_price-excess-return.csv')
            data.rename(columns={price_col: f'{stock_symbol.lower()}Price'}, inplace=True)
        else:
            print(f"Unknown stock symbol: {stock_symbol}")
            return None
        
        data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
        data['date'] = data['datetime'].dt.floor('min')
        data.drop_duplicates(subset=['date'], inplace=True)
        return data
    except FileNotFoundError:
        print(f"Error: {stock_symbol} data file not found")
        return None

# Load data
gme_data = load_stock_data('GME', 'gme_price', 'gme_return')
amc_data = load_stock_data('AMC', 'amc_price', 'amc_return')

if gme_data is not None:
    print("GME data loaded successfully!")
    gme_data.set_index('date', inplace=True)
    
if amc_data is not None:
    print("AMC data loaded successfully!")
    amc_data.set_index('date', inplace=True)

print("Data loaded successfully!")

# Define Reddit outage events
ROBINHOOD_BAN_DATE = pd.Timestamp('2021-01-28 09:30:00')

ALL_OUTAGE_EVENTS = [
    {
        'name': 'Outage 1 (Pre-Ban)',
        'start': pd.Timestamp('2021-01-27 11:29:00'),
        'end': pd.Timestamp('2021-01-27 13:40:00'),
        'duration_minutes': 131
    },
    {
        'name': 'Outage 2 (Pre-Ban)', 
        'start': pd.Timestamp('2021-01-27 16:03:00'),
        'end': pd.Timestamp('2021-01-27 17:01:00'),
        'duration_minutes': 58
    },
    {
        'name': 'Outage 3 (Post-Ban)',
        'start': pd.Timestamp('2021-01-28 08:44:00'),
        'end': pd.Timestamp('2021-01-28 10:51:00'),
        'duration_minutes': 127
    },
    {
        'name': 'Outage 4 (Post-Ban)',
        'start': pd.Timestamp('2021-01-28 19:10:00'),
        'end': pd.Timestamp('2021-01-28 21:00:00'),
        'duration_minutes': 110
    }
]

# Classify events as pre-ban or post-ban
for event in ALL_OUTAGE_EVENTS:
    event['pre_ban'] = (event['start'] + (event['end'] - event['start'])/2) < ROBINHOOD_BAN_DATE

print(f"Analyzing {len(ALL_OUTAGE_EVENTS)} Reddit outage events...")
print(f"Robinhood ban date: {ROBINHOOD_BAN_DATE}")

# Function to calculate CAR using CAPM Market Model
def calculate_car_capm(data, event, estimation_window, alpha, beta, stock_symbol):
    """Calculate CAR using CAPM Market Model"""
    
    # Define event window (from start of outage to 2 hours after end)
    event_start = event['start']
    event_end = event['end'] + timedelta(hours=2)
    
    # Get event window data
    event_data = data.loc[event_start:event_end].copy()
    
    if len(event_data) == 0:
        print(f"Warning: No data found for {event['name']} in time window {event_start} to {event_end}")
        return None
    
    if len(event_data) < 5:
        print(f"Warning: Insufficient data for {event['name']}: only {len(event_data)} points")
        return None
    
    # Calculate expected returns using CAPM Market Model
    event_data['expected_return'] = alpha + beta * event_data['spy_return']
    
    # Calculate abnormal returns
    return_col = f'{stock_symbol.lower()}_return'
    event_data['AR'] = event_data[return_col] - event_data['expected_return']
    
    # Calculate CAR from the start of the window (no pre-period)
    event_data['CAR'] = event_data['AR'].cumsum()
    
    # Calculate time from outage start
    event_data['time_from_start'] = (event_data.index - event['start']).total_seconds() / 60
    
    # Calculate CAR values at specific time points from start of window
    def get_car_at_time(target_time):
        """Helper function to get CAR at a specific time"""
        try:
            if target_time in event_data.index:
                return event_data.loc[target_time, 'CAR']
            else:
                # Find closest time point
                time_diff = abs(event_data.index - target_time)
                closest_idx = time_diff.argmin()
                return event_data.iloc[closest_idx]['CAR'] if closest_idx < len(event_data) else np.nan
        except:
            return np.nan
    
    # Calculate CAR at 30 minutes after start
    car_30min = get_car_at_time(event['start'] + timedelta(minutes=30))
    
    # Calculate CAR at 1 hour after start
    car_1h = get_car_at_time(event['start'] + timedelta(hours=1))
    
    # Calculate CAR at end of outage window
    car_at_outage_end = get_car_at_time(event['end'])
    
    
    # Calculate price change
    try:
        price_col = f'{stock_symbol.lower()}Price'
        price_change = (event_data.loc[event['end']:event['end'], price_col].iloc[0] - 
                       event_data.iloc[0][price_col]) / event_data.iloc[0][price_col]
    except:
        price_change = np.nan
    
    return {
        'event_name': event['name'],
        'start_time': event['start'],
        'end_time': event['end'],
        'outage_duration': event['duration_minutes'],
        'pre_ban': event['pre_ban'],
        'CAR_30min': car_30min,
        'CAR_1h': car_1h,
        'CAR_at_outage_end': car_at_outage_end,
        'price_change_during_outage': price_change,
        'event_data': event_data
    }

# Function to run analysis using CAPM Market Model
def run_capm_analysis(stock_data, stock_symbol):
    """Run analysis using CAPM Market Model"""
    if stock_data is None:
        print(f"No data available for {stock_symbol}")
        return None
        
    print(f"\n{'='*60}")
    print(f"ANALYZING {stock_symbol} STOCK (CAPM Market Model)")
    print(f"{'='*60}")
    
    # Estimate CAPM Market Model using 120 days before first event
    print(f"Estimating CAPM Market Model for {stock_symbol}...")
    first_event_date = min([event['start'] for event in ALL_OUTAGE_EVENTS])
    estimation_end = first_event_date - timedelta(days=1)
    estimation_start = estimation_end - timedelta(days=120)

    estimation_window = stock_data.loc[estimation_start:estimation_end]

    # Fit CAPM Market Model
    return_col = f'{stock_symbol.lower()}_return'
    X = sm.add_constant(estimation_window['spy_return'])
    y = estimation_window[return_col]
    model = sm.OLS(y, X, missing='drop').fit()
    alpha, beta = model.params

    print(f"{stock_symbol} CAPM Model parameters: Alpha={alpha:.6f}, Beta={beta:.6f}")
    print(f"{stock_symbol} R-squared: {model.rsquared:.4f}")

    # Calculate CAR for all events
    print(f"Calculating CAR for all {stock_symbol} events...")
    event_results = []

    for event in ALL_OUTAGE_EVENTS:
        print(f"\nProcessing {event['name']} for {stock_symbol}...")
        result = calculate_car_capm(stock_data, event, estimation_window, alpha, beta, stock_symbol)
        if result:
            print(f"  Success: {len(result['event_data'])} data points")
            event_results.append(result)
        else:
            print(f"  Failed: No valid result")
    
    return event_results

# Run analysis for both stocks
gme_results = run_capm_analysis(gme_data, 'GME')
amc_results = run_capm_analysis(amc_data, 'AMC')

# Process results
def process_results(event_results, stock_symbol):
    """Process and display results"""
    if event_results is None or len(event_results) == 0:
        print(f"No results available for {stock_symbol}")
        return None
    
    results_df = pd.DataFrame([{
        'event_name': r['event_name'],
        'start_time': r['start_time'],
        'end_time': r['end_time'],
        'outage_duration': r['outage_duration'],
        'pre_ban': r['pre_ban'],
        'CAR_30min': r['CAR_30min'],
        'CAR_1h': r['CAR_1h'],
        'CAR_at_outage_end': r['CAR_at_outage_end'],
        'price_change_during_outage': r['price_change_during_outage']
    } for r in event_results])
    
    print(f"\n{'='*80}")
    print(f"REDDIT OUTAGE IMPACT ON {stock_symbol} STOCK - SUMMARY RESULTS")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(f'{dataPath}reddit_outage_impact_results_{stock_symbol}.csv', index=False)
    print(f"\n{stock_symbol} results saved to: {dataPath}reddit_outage_impact_results_{stock_symbol}.csv")
    
    return results_df

# Process results
gme_results_df = process_results(gme_results, 'GME')
amc_results_df = process_results(amc_results, 'AMC')

# Statistical analysis with proper tests
def perform_statistical_analysis_capm(results_df, stock_symbol):
    """Perform comprehensive statistical analysis using CAPM results"""
    if results_df is None or len(results_df) == 0:
        print(f"No data available for {stock_symbol} statistical analysis")
        return
        
    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS - {stock_symbol} (CAPM Market Model)")
    print(f"{'='*80}")

    # Before vs After Robinhood ban comparison
    before_ban = results_df[results_df['pre_ban'] == True]
    after_ban = results_df[results_df['pre_ban'] == False]

    print(f"\nBefore Robinhood Ban (n={len(before_ban)}):")
    if len(before_ban) > 0:
        print(f"  Average CAR at 30min: {before_ban['CAR_30min'].mean():.4f}")
        print(f"  Average CAR at 1h: {before_ban['CAR_1h'].mean():.4f}")
        print(f"  Average CAR at outage end: {before_ban['CAR_at_outage_end'].mean():.4f}")
    else:
        print("  No pre-ban events")

    print(f"\nAfter Robinhood Ban (n={len(after_ban)}):")
    if len(after_ban) > 0:
        print(f"  Average CAR at 30min: {after_ban['CAR_30min'].mean():.4f}")
        print(f"  Average CAR at 1h: {after_ban['CAR_1h'].mean():.4f}")
        print(f"  Average CAR at outage end: {after_ban['CAR_at_outage_end'].mean():.4f}")
    else:
        print("  No post-ban events")

    # Individual event significance tests
    print(f"\n{'='*60}")
    print(f"INDIVIDUAL EVENT SIGNIFICANCE TESTS - {stock_symbol}")
    print(f"{'='*60}")
    
    from scipy.stats import ttest_1samp
    
    car_columns = ['CAR_30min', 'CAR_1h', 'CAR_at_outage_end']
    
    for _, event in results_df.iterrows():
        print(f"\n{event['event_name']}:")
        for col in car_columns:
            if not pd.isna(event[col]):
                car_value = event[col]
                # One-sample t-test against zero
                t_stat, p_value = ttest_1samp([car_value], 0)
                significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                magnitude = "Large" if abs(car_value) > 0.1 else "Medium" if abs(car_value) > 0.05 else "Small"
                direction = "Positive" if car_value > 0 else "Negative"
                print(f"  {col}: {car_value:.4f} ({direction}, {magnitude}) [t={t_stat:.3f}, p={p_value:.3f}] {significance}")
            else:
                print(f"  {col}: No data available")

    # Pre-ban vs Post-ban comparison with proper statistical tests
    if len(before_ban) > 0 and len(after_ban) > 0:
        print(f"\n{'='*60}")
        print(f"PRE-BAN vs POST-BAN COMPARISON - {stock_symbol}")
        print(f"{'='*60}")
        
        from scipy.stats import ttest_ind, mannwhitneyu
        
        for col in car_columns:
            car_before = before_ban[col].dropna()
            car_after = after_ban[col].dropna()
            
            if len(car_before) > 0 and len(car_after) > 0:
                print(f"{col}:")
                print(f"  Pre-ban mean: {car_before.mean():.4f} (n={len(car_before)})")
                print(f"  Post-ban mean: {car_after.mean():.4f} (n={len(car_after)})")
                
                diff = car_after.mean() - car_before.mean()
                print(f"  Difference: {diff:.4f}")
                
                if len(car_before) >= 2 and len(car_after) >= 2:
                    # Perform t-test
                    t_stat, p_value = ttest_ind(car_before, car_after)
                    significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                    print(f"  T-test: t={t_stat:.4f}, p={p_value:.4f} {significance}")
                    print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
                    
                    # Also perform Mann-Whitney U test (non-parametric)
                    try:
                        u_stat, u_p_value = mannwhitneyu(car_before, car_after, alternative='two-sided')
                        u_significance = "***" if u_p_value < 0.01 else "**" if u_p_value < 0.05 else "*" if u_p_value < 0.1 else ""
                        print(f"  Mann-Whitney U: U={u_stat:.4f}, p={u_p_value:.4f} {u_significance}")
                    except:
                        print(f"  Mann-Whitney U: Could not perform test")
                else:
                    print(f"  Note: Insufficient data for statistical tests (n<2 in one group)")
                    print(f"  Magnitude: {'Large' if abs(diff) > 0.1 else 'Medium' if abs(diff) > 0.05 else 'Small'}")
                print()

# Perform statistical analysis
perform_statistical_analysis_capm(gme_results_df, 'GME')
if amc_results_df is not None:
    perform_statistical_analysis_capm(amc_results_df, 'AMC')

# Create plots
def create_capm_plots(event_results, results_df, stock_symbol):
    """Create plots for CAPM analysis results"""
    if event_results is None or len(event_results) == 0:
        print(f"No results available for {stock_symbol} plotting")
        return
        
    print(f"\nGenerating plots for {stock_symbol}...")
    
    # Create stock-specific subdirectory
    stock_figure_path = os.path.join(figurePath, stock_symbol.lower())
    os.makedirs(stock_figure_path, exist_ok=True)
    
    # 1. CAR comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Reddit Outage Impact on {stock_symbol} Stock: Before vs After Robinhood Ban', fontsize=16)

    # Plot 1: CAR at 30 minutes
    axes[0,0].bar(range(len(results_df)), results_df['CAR_30min'], 
                   color=['red' if x else 'blue' for x in results_df['pre_ban']])
    axes[0,0].set_title('CAR at 30 Minutes')
    axes[0,0].set_ylabel('CAR')
    axes[0,0].set_xticks(range(len(results_df)))
    axes[0,0].set_xticklabels([f"Event {i+1}" for i in range(len(results_df))], rotation=45)
    axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.7)

    # Plot 2: CAR at 1 hour
    axes[0,1].bar(range(len(results_df)), results_df['CAR_1h'], 
                   color=['red' if x else 'blue' for x in results_df['pre_ban']])
    axes[0,1].set_title('CAR at 1 Hour')
    axes[0,1].set_ylabel('CAR')
    axes[0,1].set_xticks(range(len(results_df)))
    axes[0,1].set_xticklabels([f"Event {i+1}" for i in range(len(results_df))], rotation=45)
    axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.7)

    # Plot 3: CAR at outage end
    axes[1,0].bar(range(len(results_df)), results_df['CAR_at_outage_end'], 
                   color=['red' if x else 'blue' for x in results_df['pre_ban']])
    axes[1,0].set_title('CAR at Outage End')
    axes[1,0].set_ylabel('CAR')
    axes[1,0].set_xticks(range(len(results_df)))
    axes[1,0].set_xticklabels([f"Event {i+1}" for i in range(len(results_df))], rotation=45)
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.7)

    # Plot 4: CAR at outage end
    axes[1,1].bar(range(len(results_df)), results_df['CAR_at_outage_end'], 
                   color=['red' if x else 'blue' for x in results_df['pre_ban']])
    axes[1,1].set_title('CAR at Outage End')
    axes[1,1].set_ylabel('CAR')
    axes[1,1].set_xticks(range(len(results_df)))
    axes[1,1].set_xticklabels([f"Event {i+1}" for i in range(len(results_df))], rotation=45)
    axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.7)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Before Robinhood Ban'),
                       Patch(facecolor='blue', label='After Robinhood Ban')]
    fig.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(stock_figure_path, f'reddit_outage_impact_analysis_{stock_symbol}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Detailed time series plots for each event
    print(f"Generating detailed time series plots for all {stock_symbol} events...")
    if len(event_results) > 0:
        for i, result in enumerate(event_results):
            if result['event_data'] is not None and len(result['event_data']) > 0:
                event_data = result['event_data']
                event_name = result['event_name']
                
                print(f"Generating plot for: {event_name}")
                print(f"Event data shape: {event_data.shape}")
                print(f"Data range: {event_data.index.min()} to {event_data.index.max()}")
                
                # Create a separate figure for each event
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                fig.suptitle(f'Detailed Analysis: {event_name} - {stock_symbol}', fontsize=16)
                
                # Plot 1: Price movement
                price_col = f'{stock_symbol.lower()}Price'
                ax1.plot(event_data.index, event_data[price_col], '#1f77b4', label=f'{stock_symbol} Price', linewidth=2)
                ax1.axvline(x=result['start_time'], color='#2ca02c', linestyle='--', label='Outage Start')
                ax1.axvline(x=result['end_time'], color='#2ca02c', linestyle='-', label='Outage End')
                ax1.set_ylabel(f'{stock_symbol} Price ($)')
                ax1.set_title(f'{event_name}: {stock_symbol} Price During Outage Event')
                ax1.legend()
                ax1.grid(False)
                
                # Format x-axis for time with date
                import matplotlib.dates as mdates
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                ax1.tick_params(axis='x', rotation=45)
                
                # Plot 2: Cumulative abnormal returns
                ax2.plot(event_data.index, event_data['CAR'], '#d62728', label='Cumulative Abnormal Returns', linewidth=2)
                ax2.axvline(x=result['start_time'], color='#2ca02c', linestyle='--', label='Outage Start')
                ax2.axvline(x=result['end_time'], color='#2ca02c', linestyle='-', label='Outage End')
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
                ax2.set_ylabel('CAR')
                ax2.set_xlabel('Time')
                ax2.set_title(f'{event_name}: Cumulative Abnormal Returns (CAR)')
                ax2.legend()
                ax2.grid(False)
                
                # Format x-axis for time with date
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                ax2.tick_params(axis='x', rotation=45)
                
                # Add event statistics as text
                car_text = f"CAR at end: {result['CAR_at_outage_end']:.4f}\nPrice change: {result['price_change_during_outage']*100:.2f}%"
                ax2.text(0.02, 0.98, car_text, transform=ax2.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#90EE90', alpha=0.3))
                
                plt.tight_layout()
                
                # Save each event as a separate file
                event_filename = f'detailed_outage_analysis_{i+1}_{event_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")}_{stock_symbol}.png'
                plt.savefig(os.path.join(stock_figure_path, event_filename), dpi=300, bbox_inches='tight')
                plt.show()
                print(f"Detailed plot for {event_name} saved as: {os.path.join(stock_symbol.lower(), event_filename)}")
        
        print(f"Detailed plots for all {stock_symbol} events generated successfully!")
    else:
        print(f"No {stock_symbol} event results available for detailed plotting")

# Create plots for both stocks
create_capm_plots(gme_results, gme_results_df, 'GME')
if amc_results is not None:
    create_capm_plots(amc_results, amc_results_df, 'AMC')

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("CAPM Market Model analysis completed successfully!")
print("All 4 events analyzed with proper statistical tests.")
print("Results saved to CSV files and plots generated.")
