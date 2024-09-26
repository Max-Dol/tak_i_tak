import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json

def load_data():
    attributes_path = './data/train/attributes.parquet'
    resnet_path = './data/train/resnet.parquet'
    text_and_bert_path = './data/train/text_and_bert.parquet'
    train_path = './data/train/train.parquet'

    attributes = pd.read_parquet(attributes_path)
    resnet = pd.read_parquet(resnet_path)
    text_and_bert = pd.read_parquet(text_and_bert_path)
    train = pd.read_parquet(train_path)

    return attributes, resnet, text_and_bert, train

def extract_text_from_row(row):
    category_text = ' '.join(
        [' '.join(map(str, v)) if isinstance(v, list) else str(v) for v in list(row['categories'].values())]
    )
    attributes_text = ' '.join(
        [' '.join(map(str, v)) if isinstance(v, list) else str(v) for v in list(row['characteristic_attributes_mapping'].values())]
    )
    return f"{category_text} {attributes_text}"

def process_attributes(df):
    df['categories'] = df['categories'].apply(json.loads)
    df['characteristic_attributes_mapping'] = df['characteristic_attributes_mapping'].apply(json.loads)
    df['combined_text'] = df.apply(extract_text_from_row, axis=1)
    return df

def merge_data(train, resnet, attributes, text_and_bert):
    # Объединение данных с resnet
    train_data = train.merge(resnet[['variantid', 'main_pic_embeddings_resnet_v1']], left_on='variantid1', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'main_pic_embeddings_resnet_v1': 'pic_embeddings_1'})
    train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.merge(resnet[['variantid', 'main_pic_embeddings_resnet_v1']], left_on='variantid2', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'main_pic_embeddings_resnet_v1': 'pic_embeddings_2'})
    train_data = train_data.drop(columns=['variantid'])

    # Объединение данных с атрибутами
    train_data = train_data.merge(attributes[['variantid', 'combined_text']], left_on='variantid1', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'combined_text': 'text_1'})
    train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.merge(attributes[['variantid', 'combined_text']], left_on='variantid2', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'combined_text': 'text_2'})
    train_data = train_data.drop(columns=['variantid'])

    # Объединение данных с text_and_bert
    train_data = train_data.merge(text_and_bert[['variantid', 'name_bert_64']], left_on='variantid1', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'name_bert_64': 'bert_embeddings_1'})
    train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.merge(text_and_bert[['variantid', 'name_bert_64']], left_on='variantid2', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'name_bert_64': 'bert_embeddings_2'})
    train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.dropna()
    return train_data

def prepare_data(train_data, tfidf_vectorizer):
    # Объединение текстов из столбцов 'text_1' и 'text_2' для каждой строки
    text_data = train_data['text_1'] + ' ' + train_data['text_2']
    # Преобразование объединённых текстов в числовые векторы с помощью TF-IDF векторизатора
    text_embeddings = tfidf_vectorizer.fit_transform(text_data).toarray()

    # Создание столбца 'combined_embeddings' для объединённых эмбеддингов текста и изображений
    train_data['combined_embeddings'] = train_data.apply(lambda row: np.concatenate([
        row['pic_embeddings_1'][0],      # Эмбеддинги изображения из столбца 'pic_embeddings_1'
        row['pic_embeddings_2'][0],      # Эмбеддинги изображения из столбца 'pic_embeddings_2'
        text_embeddings[row.name]        # Соответствующие TF-IDF эмбеддинги для объединённого текста
    ]), axis=1)

    # Создание матрицы признаков X из объединённых эмбеддингов
    X = np.vstack(train_data['combined_embeddings'].values)
    # Метки классов (целевая переменная) из столбца 'target'
    y = train_data['target']

    # Разделение данных на тренировочную и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=23)

    # Возвращение тренировочных и валидационных данных, а также векторизатора
    return X_train, X_val, y_train, y_val, tfidf_vectorizer

def train_model(X_train, y_train):
    # Инициализация модели CatBoost
    model = CatBoostClassifier(iterations=500, 
                               learning_rate=0.1, 
                               depth=6, 
                               loss_function='Logloss', 
                               eval_metric='AUC', 
                               verbose=100)
    model.fit(X_train, y_train)
    
    # Сохранение модели в формате pkl
    joblib.dump(model, 'baseline.pkl')
    return model

def evaluate_model(model, X_val, y_val):
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    prauc = auc(recall, precision)
    print(f'PRAUC: {prauc}')

def main():
    attributes, resnet, text_and_bert, train = load_data()
    attributes = process_attributes(attributes)

    train_data = merge_data(train, resnet, attributes, text_and_bert)

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_train, X_val, y_train, y_val, tfidf_vectorizer = prepare_data(train_data, tfidf_vectorizer)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_val, y_val)
    joblib.dump(tfidf_vectorizer, 'vectorizer.pkl')

if __name__ == "__main__":
    main()
