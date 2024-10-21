import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Create mapping for A=1, B=2, C=3, D=4, E=5
letter_to_number = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}

# Function to convert OD string into numeric values
def convert_od(od_value):
    # Split OD into origin and destination
    origin = od_value[:3]  # First 3 letters are origin
    destination = od_value[3:]  # Last 3 letters are destination
    # Convert letters to numbers based on the mapping
    origin_numbers = [letter_to_number.get(char, 0) for char in origin]  # Convert each letter
    destination_numbers = [letter_to_number.get(char, 0) for char in destination]
    # Combine origin and destination into one numeric list
    return origin_numbers + destination_numbers

# Function to calculate the numeric difference between origin and destination
def compute_od_difference(od_value):
    origin = od_value[:3]
    destination = od_value[3:]
    # Subtract the total of destination from the total of origin
    origin_total = sum([letter_to_number.get(char, 0) for char in origin])
    destination_total = sum([letter_to_number.get(char, 0) for char in destination])
    return destination_total - origin_total

# Load both datasets
participant_data = pd.read_csv('participant_data.csv')
baseline_data = pd.read_csv('baseline.csv')

# Preprocessing the participant data
participant_data_clean = participant_data.dropna(subset=['choice']).copy()

# Map 'choice' column
participant_data_clean['choice'] = participant_data_clean['choice'].map({'advs': 0, 'pref': 1, 'nochoice': 2}).astype(int)

# Convert 'od' column to numeric based on letter_to_number mapping
participant_data_clean['od_numeric'] = participant_data_clean['od'].apply(convert_od)

# Create separate columns for each part of the numeric OD data
participant_data_clean[['origin_1', 'origin_2', 'origin_3', 'dest_1', 'dest_2', 'dest_3']] = pd.DataFrame(participant_data_clean['od_numeric'].tolist(), index=participant_data_clean.index)

# Compute the numeric difference between origin and destination
participant_data_clean['od_difference'] = participant_data_clean['od'].apply(compute_od_difference)

# Drop unnecessary columns but keep the new 'origin', 'destination', and 'od_difference' numeric columns
X_train = participant_data_clean.drop(columns=['id', 'ticket_id', 'flight_departure_datetime', 'purchase_datetime', 'od', 'choice', 'od_numeric'])

# Convert categorical features (branded_fare, trip_type) to dummy variables
X_train = pd.get_dummies(X_train, columns=['branded_fare', 'trip_type'], drop_first=True)

# Identify numerical columns to scale
numerical_columns = ['ADVS_price', 'PREF_price', 'ADVS_capacity', 'PREF_capacity', 'ADVS_inventory', 'PREF_inventory', 'od_difference']

# Explicitly convert numerical columns to numeric, coercing errors
for col in numerical_columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')

# Fill any NaN values that resulted from coercion
X_train[numerical_columns] = X_train[numerical_columns].fillna(0)

# Scale only the numerical columns
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])

# Target variable
y_train = participant_data_clean['choice']

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=None, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Preprocessing the baseline data for prediction
baseline_data_clean = baseline_data.copy()

# Convert datetime columns to datetime and handle invalid entries
baseline_data_clean['flight_departure_datetime'] = pd.to_datetime(baseline_data_clean['flight_departure_datetime'], errors='coerce')
baseline_data_clean['purchase_datetime'] = pd.to_datetime(baseline_data_clean['purchase_datetime'], errors='coerce')

# Drop rows where datetime conversion failed
baseline_data_clean = baseline_data_clean.dropna(subset=['flight_departure_datetime', 'purchase_datetime'])

# Convert 'od' column in baseline data to numeric using the same method
baseline_data_clean['od_numeric'] = baseline_data_clean['od'].apply(convert_od)

# Create separate columns for each part of the numeric OD data in the baseline data
baseline_data_clean[['origin_1', 'origin_2', 'origin_3', 'dest_1', 'dest_2', 'dest_3']] = pd.DataFrame(baseline_data_clean['od_numeric'].tolist(), index=baseline_data_clean.index)

# Compute the numeric difference for baseline data
baseline_data_clean['od_difference'] = baseline_data_clean['od'].apply(compute_od_difference)

# Drop columns that shouldn't be part of the prediction data
X_baseline = baseline_data_clean.drop(columns=['id', 'ticket_id', 'flight_departure_datetime', 'purchase_datetime', 'od', 'choice', 'od_numeric'])

# Convert categorical features to dummy variables (ensure 'branded_fare' and 'trip_type' are included)
X_baseline = pd.get_dummies(X_baseline, columns=['branded_fare', 'trip_type'], drop_first=True)

# Explicitly convert numerical columns in baseline_data to numeric, coercing errors
for col in numerical_columns:
    X_baseline[col] = pd.to_numeric(X_baseline[col], errors='coerce')

# Fill any NaN values in baseline_data caused by coercion
X_baseline[numerical_columns] = X_baseline[numerical_columns].fillna(0)

# Scale only the numerical columns in baseline data
X_baseline[numerical_columns] = scaler.transform(X_baseline[numerical_columns])

# Align the columns of the baseline data with the training data (fill missing columns with 0)
X_baseline = X_baseline.reindex(columns=X_train.columns, fill_value=0)

# Predict seat selection on the baseline data using the trained model
baseline_data_clean['choice'] = model.predict(X_baseline)

# Map numerical predictions back to string labels
baseline_data_clean['choice'] = baseline_data_clean['choice'].map({0: 'advs', 1: 'pref', 2: 'nochoice'})

# Now, keep only the columns you want for the final output
final_columns = ['id', 'ticket_id', 'od', 'flight_departure_datetime', 'purchase_datetime', 'trip_type', 'branded_fare',
                 'number_of_pax', 'ADVS_price', 'PREF_price', 'ADVS_capacity', 'PREF_capacity', 
                 'ADVS_inventory', 'PREF_inventory', 'choice']

baseline_data_final = baseline_data_clean[final_columns]

# Save the updated baseline data with predictions
baseline_data_final.to_csv('9360.csv', index=False)

# Display a preview of the predictions
print(baseline_data_final.head())