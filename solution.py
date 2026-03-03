from __future__ import annotations

import gzip
import html
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import LinearSVC

TEXT_COLUMNS = [
    "vendor_name",
    "vendor_code",
    "title",
    "description",
    "shop_category_name",
]
OVERRIDE_RULES = [
    ("vendor_name", "title"),
    ("shop_category_name",),
    ("shop_category_name", "title"),
    ("title",),
    ("vendor_code",),
]
WORD_VECTORIZER_PARAMS = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "max_features": 60_000,
    "dtype": np.float32,
}
CHAR_VECTORIZER_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "sublinear_tf": True,
    "max_features": 70_000,
    "dtype": np.float32,
}
DEPARTMENT_MODEL_PARAMS = {
    "class_weight": "balanced",
    "C": 1.0,
    "random_state": 42,
}
BATCH_SIZE = 256


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset = pd.read_csv(path, sep="\t")
    for column in TEXT_COLUMNS:
        if column not in dataset.columns:
            dataset[column] = ""
    return dataset


def build_search_texts(dataframe: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    rows = dataframe[TEXT_COLUMNS].itertuples(index=False, name=None)
    for vendor_name, vendor_code, title, description, shop_category_name in rows:
        parts = [
            "vendor " + clean_text(vendor_name),
            "code " + clean_text(vendor_code),
            "title " + clean_text(title),
            "title " + clean_text(title),
            "shopcat " + clean_text(shop_category_name),
            "shopcat " + clean_text(shop_category_name),
            "desc " + clean_text(description),
        ]
        texts.append(" ".join(parts))
    return texts


def build_rule_keys(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    key_parts = [dataframe[column].map(clean_text) for column in columns]
    key = key_parts[0]
    for part in key_parts[1:]:
        key = key + "||" + part
    return key.to_numpy(dtype=object)


def has_meaningful_key(key: object) -> bool:
    return str(key).strip("|") != ""


def build_unique_category_map(
    dataframe: pd.DataFrame, columns: tuple[str, ...]
) -> dict[str, int]:
    temp = dataframe[list(columns) + ["category_id"]].copy()
    temp["key"] = build_rule_keys(temp, columns)
    temp = temp[temp["key"].map(has_meaningful_key)]
    counts = temp.groupby("key")["category_id"].nunique()
    unique_keys = counts[counts == 1].index
    if len(unique_keys) == 0:
        return {}
    unique_rows = temp[temp["key"].isin(unique_keys)].drop_duplicates("key")
    return dict(zip(unique_rows["key"], unique_rows["category_id"]))


def fit_feature_blocks(
    texts: list[str],
) -> tuple[TfidfVectorizer, TfidfVectorizer, sparse.csr_matrix]:
    word_vectorizer = TfidfVectorizer(**WORD_VECTORIZER_PARAMS)
    char_vectorizer = TfidfVectorizer(**CHAR_VECTORIZER_PARAMS)
    word_features = word_vectorizer.fit_transform(texts)
    char_features = char_vectorizer.fit_transform(texts)
    features = sparse.hstack(
        [word_features, char_features],
        format="csr",
        dtype=np.float32,
    )
    return word_vectorizer, char_vectorizer, features


def transform_feature_blocks(
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    texts: list[str],
) -> sparse.csr_matrix:
    word_features = word_vectorizer.transform(texts)
    char_features = char_vectorizer.transform(texts)
    return sparse.hstack(
        [word_features, char_features],
        format="csr",
        dtype=np.float32,
    )


def fit_bundle(dataframe: pd.DataFrame) -> dict[str, object]:
    texts = build_search_texts(dataframe)
    word_vectorizer, char_vectorizer, train_matrix = fit_feature_blocks(texts)

    department_model = LinearSVC(**DEPARTMENT_MODEL_PARAMS)
    department_model.fit(train_matrix, dataframe["department_id"])

    override_maps = []
    for columns in OVERRIDE_RULES:
        override_maps.append(
            {
                "columns": columns,
                "mapping": build_unique_category_map(dataframe, columns),
            }
        )

    return {
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "department_model": department_model,
        "train_matrix": train_matrix,
        "train_categories": dataframe["category_id"].to_numpy(dtype=np.int32),
        "override_maps": override_maps,
    }


def nearest_neighbor_categories(
    query_matrix: sparse.csr_matrix,
    train_matrix: sparse.csr_matrix,
    train_categories: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    predictions = np.empty(query_matrix.shape[0], dtype=np.int64)
    for start in range(0, query_matrix.shape[0], batch_size):
        stop = min(start + batch_size, query_matrix.shape[0])
        scores = query_matrix[start:stop] @ train_matrix.T
        best_indices = np.asarray(scores.argmax(axis=1)).ravel()
        predictions[start:stop] = train_categories[best_indices]
    return predictions


def apply_category_overrides(
    dataframe: pd.DataFrame,
    base_categories: np.ndarray,
    override_maps: list[dict[str, object]],
) -> np.ndarray:
    predictions = base_categories.copy()
    if len(predictions) == 0:
        return predictions

    rule_keys = []
    for rule in override_maps:
        columns = rule["columns"]
        rule_keys.append(build_rule_keys(dataframe, columns))

    for row_index in range(len(predictions)):
        for rule_index, rule in enumerate(override_maps):
            mapping = rule["mapping"]
            key = rule_keys[rule_index][row_index]
            if has_meaningful_key(key) and key in mapping:
                predictions[row_index] = mapping[key]
                break
    return predictions


def predict_with_bundle(
    model_bundle: dict[str, object], dataframe: pd.DataFrame
) -> pd.DataFrame:
    texts = build_search_texts(dataframe)
    query_matrix = transform_feature_blocks(
        model_bundle["word_vectorizer"],
        model_bundle["char_vectorizer"],
        texts,
    )

    department_predictions = model_bundle["department_model"].predict(query_matrix)
    base_categories = nearest_neighbor_categories(
        query_matrix,
        model_bundle["train_matrix"],
        model_bundle["train_categories"],
    )
    category_predictions = apply_category_overrides(
        dataframe,
        base_categories,
        model_bundle["override_maps"],
    )

    return pd.DataFrame(
        {
            "department_id": department_predictions.astype(np.int64),
            "category_id": category_predictions.astype(np.int64),
        }
    )


def score_predictions(
    truth: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, float]:
    department_f1 = f1_score(
        truth["department_id"],
        predictions["department_id"],
        average="macro",
    )
    category_accuracy = accuracy_score(
        truth["category_id"],
        predictions["category_id"],
    )
    score = 30.0 * department_f1 + 70.0 * category_accuracy
    return {
        "score": score,
        "department_f1": department_f1,
        "category_accuracy": category_accuracy,
    }


def evaluate_with_splits(
    dataframe: pd.DataFrame,
    n_splits: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    splitter = ShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    scores = []
    for train_indices, valid_indices in splitter.split(dataframe):
        train_fold = dataframe.iloc[train_indices].reset_index(drop=True)
        valid_fold = dataframe.iloc[valid_indices].reset_index(drop=True)
        model_bundle = fit_bundle(train_fold)
        predictions = predict_with_bundle(model_bundle, valid_fold)
        scores.append(score_predictions(valid_fold, predictions))

    metrics = {
        "score_mean": float(np.mean([item["score"] for item in scores])),
        "score_std": float(np.std([item["score"] for item in scores])),
        "department_f1_mean": float(
            np.mean([item["department_f1"] for item in scores])
        ),
        "category_accuracy_mean": float(
            np.mean([item["category_accuracy"] for item in scores])
        ),
    }
    return metrics


def save_bundle(model_bundle: dict[str, object], path: str | Path) -> None:
    with gzip.open(path, "wb") as file:
        pickle.dump(model_bundle, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(path: str | Path) -> dict[str, object]:
    with gzip.open(path, "rb") as file:
        return pickle.load(file)


def predict_to_csv(
    model_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    model_bundle = load_bundle(model_path)
    test = load_dataset(test_path)
    predictions = predict_with_bundle(model_bundle, test)
    predictions.to_csv(output_path, index=False)
    return predictions
