# Predictive Modeling for Airline Seat Selection

This project develops a predictive model to forecast Air Canada customers' seat choices (advs, pref, or nochoice) based on factors like ticket pricing, seat availability, and trip details. The model was part of a machine learning and data science challenge during the CodeML Hackathon using only the provided datasets.

## Installation

### Prerequisites
- Python 3.7+
- Libraries: pandas, scikit-learn, joblib

### Setup
1. Clone the repository and navigate to the project folder.
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `participant_data.csv` and `baseline.csv` are in the project directory.

## Usage

1. Run the script to train the model and make predictions:
   ```bash
   python main.py
   ```
   - Outputs a CSV file (`9360.csv`) with predicted seat choices.

## Model Description

- **Random Forest Classifier** for seat prediction based on features like:
  - Days to Departure
  - Price and Capacity Differences
  - OD Differences

## Results

The model predicts seat choices accurately based on key features. Outputs are saved in `9360.csv`.

## Future Improvements

- Explore more advanced feature engineering.
- Test additional algorithms (e.g., XGBoost).

---
