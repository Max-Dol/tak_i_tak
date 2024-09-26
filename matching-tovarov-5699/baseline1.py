import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json

def load_data(sample_size=1):
    attributes = pd.read_parquet('./data/train/attributes.parquet')
    resnet = pd.read_parquet('./data/train/resnet.parquet')
    text_and_bert = pd.read_parquet('./data/train/text_and_bert.parquet')
    train = pd.read_parquet('./data/train/train.parquet')
    

    if sample_size < 1:
        attributes = attributes.sample(frac=sample_size, random_state=42)
        resnet = resnet.sample(frac=sample_size, random_state=42)
        text_and_bert = text_and_bert.sample(frac=sample_size, random_state=42)
        train = train.sample(frac=sample_size, random_state=42)
        

    return attributes, resnet, text_and_bert, train, 
    


def extract_text_from_row(row):

    category_text = ' '.join(
        [' '.join(map(str, v)) if isinstance(v, list) else str(v) for v in list(row['categories'].values())]
    )
    attributes_text = ' '.join(
        [' '.join(map(str, v)) if isinstance(v, list) else str(v) for v in list(row['characteristic_attributes_mapping'].values())]
    )
    return f"{category_text} {attributes_text}"

def process_text_and_bert(df):
    df['categories'] = df['categories'].apply(json.loads)
    df['characteristic_attributes_mapping'] = df['characteristic_attributes_mapping'].apply(json.loads)
    df['combined_text'] = df.apply(extract_text_from_row, axis=1)
    return df

def merge_data(train, resnet, attributes):
    train_data = train.merge(resnet[['variantid', 'main_pic_embeddings_resnet_v1']], left_on='variantid1', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'main_pic_embeddings_resnet_v1': 'pic_embeddings_1'})
    train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.merge(resnet[['variantid', 'main_pic_embeddings_resnet_v1']], left_on='variantid2', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'main_pic_embeddings_resnet_v1': 'pic_embeddings_2'})
    train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.merge(attributes[['variantid', 'combined_text']], left_on='variantid1', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'combined_text': 'text_1'})
    train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.merge(attributes[['variantid', 'combined_text']], left_on='variantid2', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'combined_text': 'text_2'})
    train_data = train_data.drop(columns=['variantid'])

    #     # Объединение с текстом из attributes для variantid1 и variantid2
    # train_data = train_data.merge(attributes[['variantid', 'combined_text']], left_on='variantid1', right_on='variantid', how='left')
    # train_data = train_data.rename(columns={'combined_text': 'text_1'})
    # train_data = train_data.drop(columns=['variantid'])

    # train_data = train_data.merge(attributes[['variantid', 'combined_text']], left_on='variantid2', right_on='variantid', how='left')
    # train_data = train_data.rename(columns={'combined_text': 'text_2'})
    # train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.dropna()

    return train_data

def combine_embeddings(row):
    # Объединение эмбеддингов изображений из столбцов 'pic_embeddings_1' и 'pic_embeddings_2'
    pic_embeddings = np.concatenate([row['pic_embeddings_1'][0], row['pic_embeddings_2'][0]])
    # Объединение эмбеддингов текста из столбцов 'text_embedding_1' и 'text_embedding_2'
    text_embeddings = np.concatenate([row['text_embedding_1'], row['text_embedding_2']])
    # Возвращение объединённых эмбеддингов изображений и текста
    return np.concatenate([pic_embeddings, text_embeddings])

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
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
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
    attributes = process_text_and_bert(attributes)

    train_data = merge_data(train, resnet, attributes)

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_train, X_val, y_train, y_val, tfidf_vectorizer = prepare_data(train_data, tfidf_vectorizer)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_val, y_val)
    joblib.dump(tfidf_vectorizer, 'vectorizer.pkl')

if __name__ == "__main__":
    main()