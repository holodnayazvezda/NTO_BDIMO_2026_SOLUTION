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
    "max_features": 100_000,
    "dtype": np.float32,
}
CHAR_VECTORIZER_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "sublinear_tf": True,
    "max_features": 100_000,
    "dtype": np.float32,
}
DEPARTMENT_MODEL_PARAMS = {
    "class_weight": "balanced",
    "C": 0.4,
    "random_state": 42,
}
BATCH_SIZE = 256
VENDOR_REPEATS = 1
CODE_REPEATS = 1
TITLE_REPEATS = 5
SHOP_CATEGORY_REPEATS = 5
DESCRIPTION_REPEATS = 1
UNCERTAIN_MARGIN_THRESHOLD = 0.25
UNCERTAIN_TOP_DEPARTMENTS = 3
_TITLE_PUNCT_RE = re.compile(r"[^0-9a-zа-я]+", re.IGNORECASE)
_TITLE_NUM_RE = re.compile(r"\d+")


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def canonicalize_title(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = _TITLE_PUNCT_RE.sub(" ", text)
    text = _TITLE_NUM_RE.sub("#", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
        c_vendor = clean_text(vendor_name)
        c_code = clean_text(vendor_code)
        c_title = clean_text(title)
        c_desc = clean_text(description)
        c_shopcat = clean_text(shop_category_name)

        parts = []
        for _ in range(VENDOR_REPEATS):
            parts.append("vendor " + c_vendor)
        for _ in range(CODE_REPEATS):
            parts.append("code " + c_code)
        for _ in range(TITLE_REPEATS):
            parts.append("title " + c_title)
        for _ in range(SHOP_CATEGORY_REPEATS):
            parts.append("shopcat " + c_shopcat)
        for _ in range(DESCRIPTION_REPEATS):
            parts.append("desc " + c_desc)

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


def build_unique_title_norm_map(dataframe: pd.DataFrame) -> dict[str, int]:
    temp = dataframe[["title", "category_id"]].copy()
    temp["key"] = temp["title"].map(canonicalize_title)
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

    category_department_map = (
        dataframe.drop_duplicates("category_id")
        .set_index("category_id")["department_id"]
        .astype(np.int64)
        .to_dict()
    )

    override_maps = []
    for columns in OVERRIDE_RULES:
        override_maps.append(
            {
                "columns": columns,
                "mapping": build_unique_category_map(dataframe, columns),
            }
        )
    title_norm_map = build_unique_title_norm_map(dataframe)

    return {
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "department_model": department_model,
        "train_matrix": train_matrix,
        "train_categories": dataframe["category_id"].to_numpy(dtype=np.int32),
        "train_departments": dataframe["department_id"].to_numpy(dtype=np.int16),
        "category_department_map": category_department_map,
        "override_maps": override_maps,
        "title_norm_map": title_norm_map,
    }


def nearest_neighbor_categories(
    query_matrix: sparse.csr_matrix,
    train_matrix: sparse.csr_matrix,
    train_categories: np.ndarray,
    train_departments: np.ndarray,
    top_departments: np.ndarray,
    uncertain_mask: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    predictions = np.empty(query_matrix.shape[0], dtype=np.int64)
    all_indices = np.arange(train_categories.shape[0], dtype=np.int32)
    department_train_indices = {
        int(department): np.flatnonzero(train_departments == department)
        for department in np.unique(train_departments)
    }

    predicted_departments = top_departments[:, 0]
    confident_query_indices = np.flatnonzero(~uncertain_mask)

    for department in np.unique(predicted_departments[confident_query_indices]):
        query_indices = confident_query_indices[
            predicted_departments[confident_query_indices] == department
        ]
        if len(query_indices) == 0:
            continue

        candidate_indices = department_train_indices.get(int(department), all_indices)
        candidate_matrix = train_matrix[candidate_indices]

        for start in range(0, len(query_indices), batch_size):
            stop = min(start + batch_size, len(query_indices))
            batch_query_indices = query_indices[start:stop]
            scores = query_matrix[batch_query_indices] @ candidate_matrix.T
            best_local_indices = np.asarray(scores.argmax(axis=1)).ravel()
            best_indices = candidate_indices[best_local_indices]
            predictions[batch_query_indices] = train_categories[best_indices]

    for query_index in np.flatnonzero(uncertain_mask):
        candidate_parts = []
        for department in top_departments[query_index]:
            candidate_parts.append(
                department_train_indices.get(
                    int(department),
                    np.empty(0, dtype=np.int32),
                )
            )
        candidate_indices = np.concatenate(candidate_parts) if candidate_parts else all_indices
        if len(candidate_indices) == 0:
            candidate_indices = all_indices
        candidate_matrix = train_matrix[candidate_indices]
        scores = query_matrix[query_index : query_index + 1] @ candidate_matrix.T
        best_local_index = int(np.asarray(scores.argmax(axis=1)).ravel()[0])
        predictions[query_index] = train_categories[candidate_indices[best_local_index]]
    return predictions


def apply_category_overrides_with_flags(
    dataframe: pd.DataFrame,
    base_categories: np.ndarray,
    override_maps: list[dict[str, object]],
    title_norm_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    predictions = base_categories.copy()
    override_used = np.zeros(len(predictions), dtype=bool)
    if len(predictions) == 0:
        return predictions, override_used

    title_norm_keys = dataframe["title"].map(canonicalize_title).to_numpy(dtype=object)

    rule_keys = []
    for rule in override_maps:
        columns = rule["columns"]
        rule_keys.append(build_rule_keys(dataframe, columns))

    for row_index in range(len(predictions)):
        key_norm = title_norm_keys[row_index]
        if has_meaningful_key(key_norm) and key_norm in title_norm_map:
            predictions[row_index] = title_norm_map[key_norm]
            override_used[row_index] = True
            continue
        for rule_index, rule in enumerate(override_maps):
            mapping = rule["mapping"]
            key = rule_keys[rule_index][row_index]
            if has_meaningful_key(key) and key in mapping:
                predictions[row_index] = mapping[key]
                override_used[row_index] = True
                break
    return predictions, override_used


def apply_category_overrides(
    dataframe: pd.DataFrame,
    base_categories: np.ndarray,
    override_maps: list[dict[str, object]],
    title_norm_map: dict[str, int],
) -> np.ndarray:
    predictions, _ = apply_category_overrides_with_flags(
        dataframe,
        base_categories,
        override_maps,
        title_norm_map,
    )
    return predictions


def get_department_candidates(
    department_model: LinearSVC,
    query_matrix: sparse.csr_matrix,
    top_k: int = UNCERTAIN_TOP_DEPARTMENTS,
) -> tuple[np.ndarray, np.ndarray]:
    decision_scores = department_model.decision_function(query_matrix)
    if decision_scores.ndim == 1:
        top_departments = department_model.predict(query_matrix).astype(np.int64).reshape(-1, 1)
        uncertain_mask = np.zeros(query_matrix.shape[0], dtype=bool)
        return top_departments, uncertain_mask

    top_k = max(1, min(top_k, decision_scores.shape[1]))
    sorted_indices = np.argsort(decision_scores, axis=1)
    top_indices = sorted_indices[:, -top_k:][:, ::-1]
    top_departments = np.asarray(department_model.classes_[top_indices], dtype=np.int64)

    top_scores = decision_scores[np.arange(decision_scores.shape[0]), top_indices[:, 0]]
    second_scores = decision_scores[np.arange(decision_scores.shape[0]), top_indices[:, 1]]
    uncertain_mask = (top_scores - second_scores) < UNCERTAIN_MARGIN_THRESHOLD
    return top_departments, uncertain_mask


def predict_with_bundle(
    model_bundle: dict[str, object], dataframe: pd.DataFrame
) -> pd.DataFrame:
    texts = build_search_texts(dataframe)
    query_matrix = transform_feature_blocks(
        model_bundle["word_vectorizer"],
        model_bundle["char_vectorizer"],
        texts,
    )

    top_departments, uncertain_mask = get_department_candidates(
        model_bundle["department_model"],
        query_matrix,
    )
    department_predictions = top_departments[:, 0]
    base_categories = nearest_neighbor_categories(
        query_matrix,
        model_bundle["train_matrix"],
        model_bundle["train_categories"],
        model_bundle["train_departments"],
        top_departments,
        uncertain_mask,
    )
    category_predictions, override_used = apply_category_overrides_with_flags(
        dataframe,
        base_categories,
        model_bundle["override_maps"],
        model_bundle["title_norm_map"],
    )
    final_departments = department_predictions.astype(np.int64, copy=True)
    category_department_map = model_bundle["category_department_map"]
    override_indices = np.flatnonzero(override_used)
    if len(override_indices) > 0:
        final_departments[override_indices] = np.array(
            [
                category_department_map.get(
                    int(category_predictions[index]),
                    int(final_departments[index]),
                )
                for index in override_indices
            ],
            dtype=np.int64,
        )

    return pd.DataFrame(
        {
            "department_id": final_departments,
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


if __name__ == "__main__":
    df = load_dataset("train.tsv")
    print(evaluate_with_splits(df))
