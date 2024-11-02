# modules/predict.py

import os
import pandas as pd
import dill
import json

def load_model(model_path: str):
    with open(model_path, 'rb') as file:
        model = dill.load(file)
    return model

def make_predictions(model, test_data: pd.DataFrame) -> pd.Series:
    """Make predictions using the loaded model on the test data."""
    predictions = model.predict(test_data)
    return predictions

def save_predictions(predictions: pd.Series, output_path: str):
    """Save predictions to a CSV file."""
    predictions_df = pd.DataFrame(predictions, columns=['predictions'])
    predictions_df.to_csv(output_path, index=False)

def predict():
    """Main function to load model, make predictions, and save them."""
    # Указываем путь к файлам проекта
    path = os.environ.get('PROJECT_PATH', '.')

    # Загружаем обученную модель
    model_path = os.path.join(path, 'data/models/cars_pipe_202411021256.pkl')  
    model = load_model(model_path)

    # Загружаем тестовые данные
    test_data_path = os.path.join(path, 'data/test')
    test_files = [f for f in os.listdir(test_data_path) if f.endswith('.json')]
    
    # Читаем тестовые данные и объединяем их в один DataFrame
    test_data_frames = []
    for file in test_files:
        file_path = os.path.join(test_data_path, file)
        print(f"Loading data from {file_path}")
        
        # Load data from JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Wrap data in a list to convert to DataFrame
        df = pd.DataFrame([data])  # Make sure to wrap the dict in a list
        
        # Print DataFrame shape and columns
        print(f"Data from {file_path} loaded successfully. Shape: {df.shape}, Columns: {df.columns.tolist()}")
        
        # Добавляем только загруженные DataFrame в список
        test_data_frames.append(df)

    # Объединяем все тестовые данные в один DataFrame
    if test_data_frames:
        test_data = pd.concat(test_data_frames, ignore_index=True)
        print(f"Combined test data shape: {test_data.shape}")
        
        # Выполняем предсказания
        predictions = make_predictions(model, test_data)

        # Сохраняем предсказания в CSV
        output_path = os.path.join(path, 'data/predictions/predictions.csv')
        save_predictions(predictions, output_path)
    else:
        print("No test data to process.")

if __name__ == '__main__':
    predict()
