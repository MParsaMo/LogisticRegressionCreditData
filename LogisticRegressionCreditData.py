import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import os

def load_data(file_path):
    """
    Loads credit data from a CSV file into a Pandas DataFrame.
    If the file is not found, a dummy CSV is created for demonstration.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame, or None if an error occurs
                          and a dummy file cannot be created.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure 'credit_data.csv' is in the same directory as the script.")
        print("Creating a dummy 'credit_data.csv' for demonstration purposes.")
        # Create a dummy CSV for demonstration
        dummy_data = {
            'income': [50000, 60000, 30000, 70000, 45000, 55000, 65000, 25000, 80000, 40000],
            'age': [30, 45, 22, 50, 35, 28, 40, 60, 33, 25],
            'loan': [10000, 20000, 5000, 30000, 8000, 15000, 25000, 3000, 40000, 7000],
            'default': [0, 0, 1, 0, 0, 0, 0, 1, 0, 1] # 0 = no default, 1 = default
        }
        pd.DataFrame(dummy_data).to_csv(file_path, index=False)
        print("Dummy 'credit_data.csv' created. Please replace it with your actual data.")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

def explore_data(dataframe):
    """
    Performs initial data analysis by printing head, descriptive statistics,
    and correlation matrix.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame.
    """
    print("\n--- Data Head ---")
    print(dataframe.head())
    print("\n--- Data Description ---")
    print(dataframe.describe())
    print("\n--- Correlation Matrix ---")
    print(dataframe.corr())

def prepare_features_target(dataframe, feature_cols, target_col):
    """
    Separates the DataFrame into features (X) and target (y).

    Args:
        dataframe (pandas.DataFrame): The input DataFrame.
        feature_cols (list): A list of column names to be used as features.
        target_col (str): The name of the target column.

    Returns:
        tuple: A tuple containing (features (X), target (y)).
    """
    features = dataframe[feature_cols]
    target = dataframe[target_col]
    return features, target

def split_data(features, target, test_size=0.3, random_state=None):
    """
    Splits the dataset into training and testing sets.

    Args:
        features (pandas.DataFrame): The feature data.
        target (pandas.Series): The target data.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split.
                                      Pass an int for reproducible output across multiple function calls.

    Returns:
        tuple: A tuple containing (feature_train, feature_test, target_train, target_test).
    """
    return train_test_split(features, target, test_size=test_size, random_state=random_state)

def train_model(feature_train, target_train):
    """
    Trains a Logistic Regression model.

    Args:
        feature_train (pandas.DataFrame): Training features.
        target_train (pandas.Series): Training target.

    Returns:
        sklearn.linear_model.LogisticRegression: The trained Logistic Regression model.
    """
    model = LogisticRegression(solver='liblinear', random_state=0) # 'liblinear' is good for small datasets
    model_fit = model.fit(feature_train, target_train)
    return model_fit

def evaluate_model(model_fit, feature_test, target_test):
    """
    Evaluates the trained Logistic Regression model and prints metrics.

    Args:
        model_fit (sklearn.linear_model.LogisticRegression): The trained model.
        feature_test (pandas.DataFrame): Testing features.
        target_test (pandas.Series): Testing target.
    """
    prediction = model_fit.predict(feature_test)

    print("\n--- Model Coefficients and Intercept ---")
    # Intercept (b0): The log-odds of the target when all features are zero.
    print(f'Intercept (b0): {model_fit.intercept_[0]:.4f}')
    # Coefficients (b1, b2, b3...): The change in the log-odds of the target
    # for a one-unit increase in the corresponding feature, holding others constant.
    print(f'Coefficients (b1, b2, b3 for income, age, loan respectively): {model_fit.coef_[0]}')

    print("\n--- Model Evaluation ---")
    # Confusion Matrix: A table used to describe the performance of a classification model
    # on a set of test data for which the true values are known.
    # It shows the number of True Positives, True Negatives, False Positives, and False Negatives.
    print('Confusion Matrix:')
    print(confusion_matrix(target_test, prediction))

    # Accuracy Score: The proportion of correctly classified instances (true positives + true negatives)
    # out of the total number of instances.
    print('Accuracy Score:')
    print(accuracy_score(target_test, prediction))

if __name__ == "__main__":
    CSV_FILE_PATH = 'credit_data.csv'
    FEATURE_COLUMNS = ['income', 'age', 'loan']
    TARGET_COLUMN = 'default'
    TEST_DATA_SPLIT_RATIO = 0.3
    RANDOM_STATE_FOR_SPLIT = 42 # For reproducibility

    # 1. Load Data
    credit_data_df = load_data(CSV_FILE_PATH)
    if credit_data_df is None:
        exit() # Exit if data loading failed

    # 2. Explore Data
    explore_data(credit_data_df)

    # 3. Prepare Features and Target
    features_df, target_series = prepare_features_target(credit_data_df, FEATURE_COLUMNS, TARGET_COLUMN)

    # 4. Split Data into Training and Testing Sets
    feature_train_set, feature_test_set, target_train_set, target_test_set = split_data(
        features_df, target_series, test_size=TEST_DATA_SPLIT_RATIO, random_state=RANDOM_STATE_FOR_SPLIT
    )
    print(f"\nData split: {len(feature_train_set)} training samples, {len(feature_test_set)} testing samples.")

    # 5. Train Model
    logistic_model = train_model(feature_train_set, target_train_set)

    # 6. Evaluate Model
    evaluate_model(logistic_model, feature_test_set, target_test_set)
