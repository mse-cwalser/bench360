import pandas as pd

# Load the raw data for calculations
emissions_df = pd.read_csv('emissions.csv')
predictions_df = pd.read_csv('vrdu_predictions.csv')

# Extract the last entry (row) from the emissions dataframe
last_emission = emissions_df.iloc[-1]

# Calculate target metrics using only the last emission entry
total_energy_wh = last_emission['energy_consumed'] * 1000  # Convert kWh to Wh
avg_field_em = predictions_df['field_em'].mean()
avg_field_f1 = predictions_df['field_f1'].mean()

# Calculate additional available metrics from the last emission entry
total_generation_time_s = last_emission['duration']
num_queries = len(predictions_df)
avg_power_w = last_emission[['cpu_power', 'gpu_power', 'ram_power']].sum()
avg_cpu_util_pct = last_emission['cpu_utilization_percent']
avg_gpu_util_pct = last_emission['gpu_utilization_percent']
avg_ram_mb = last_emission['ram_used_gb'] * 1024

# Create a dictionary for the row, explicitly defining the columns we care about
new_row = {
    'model_name': 'NuExtract',
    'task': 'vrdu_registration',
    'num_queries': num_queries,
    'total_generation_time_s': total_generation_time_s,
    'avg_power_w': avg_power_w,
    'total_energy_wh': total_energy_wh,
    'avg_cpu_util_pct': avg_cpu_util_pct,
    'avg_gpu_util_pct': avg_gpu_util_pct,
    'avg_ram_mb': avg_ram_mb,
    'avg_field_f1': avg_field_f1,
    'avg_field_em': avg_field_em
}

# Create the final dataframe
output_df = pd.DataFrame([new_row])
output_filename = 'nuextract_vrdu_metrics.csv'

# Save to the new CSV file
output_df.to_csv(output_filename, index=False)
print(f"Results successfully saved to '{output_filename}'.")