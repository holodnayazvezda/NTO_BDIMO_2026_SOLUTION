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
USE_NORMALIZED_VENDOR_CODE_OVERRIDES = True
VENDOR_CODE_NORM_MIN_LEN = 5
_VENDOR_CODE_NORM_RE = re.compile(r"[^0-9a-zа-я]+", re.IGNORECASE)
OVERRIDE_RULES = [
    ("shop_category_name", "title"),
    ("vendor_name", "title"),
    ("title",),
    ("vendor_code",),
    ("vendor_name", "vendor_code"),
    ("shop_category_name", "vendor_code"),
    ("shop_category_name",),
]
AMBIGUOUS_OVERRIDE_RULES = [
    {
        "columns": ("vendor_code",),
        "store_max_categories": 5,
        "apply_max_categories": 5,
        "min_gain_over_base": 0.0,
    },
    {
        "columns": ("vendor_name", "vendor_code"),
        "store_max_categories": 5,
        "apply_max_categories": 5,
        "min_gain_over_base": 0.0,
    },
    {
        "columns": ("shop_category_name", "vendor_code"),
        "store_max_categories": 10,
        "apply_max_categories": 10,
        "min_gain_over_base": 0.0,
    },
    {
        "columns": ("shop_category_name", "title"),
        "store_max_categories": 5,
        "apply_max_categories": 5,
        "min_gain_over_base": 0.01,
    },
    {
        "columns": ("vendor_name", "title"),
        "store_max_categories": 5,
        "apply_max_categories": 5,
        "min_gain_over_base": 0.01,
    },
    {
        "columns": ("title",),
        "store_max_categories": 5,
        "apply_max_categories": 5,
        "min_gain_over_base": 0.02,
    },
    {
        "columns": ("shop_category_name",),
        "store_max_categories": 50,
        "apply_max_categories": 15,
        "min_gain_over_base": 0.03,
    },
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
TITLE_WORD_VECTORIZER_PARAMS = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "max_features": 100_000,
    "dtype": np.float32,
}
TITLE_CHAR_VECTORIZER_PARAMS = {
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
TITLE_BONUS_WEIGHT = 0.2
DEPARTMENT_SYNC_SCORE_GAP_THRESHOLD = 0.5


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_vendor_code(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = _VENDOR_CODE_NORM_RE.sub("", text)
    if len(text) < VENDOR_CODE_NORM_MIN_LEN:
        return ""
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


def build_title_texts(dataframe: pd.DataFrame) -> list[str]:
    return dataframe["title"].map(clean_text).tolist()


def build_rule_keys(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    key_parts = [dataframe[column].map(clean_text) for column in columns]
    key = key_parts[0]
    for part in key_parts[1:]:
        key = key + "||" + part
    return key.to_numpy(dtype=object)


def build_rule_keys_vendor_code_normalized(
    dataframe: pd.DataFrame, columns: tuple[str, ...]
) -> np.ndarray:
    key_parts = []
    for column in columns:
        if column == "vendor_code":
            key_parts.append(dataframe[column].map(normalize_vendor_code))
        else:
            key_parts.append(dataframe[column].map(clean_text))
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


def build_unique_category_map_with_vendor_code_norm(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...],
) -> dict[str, int]:
    temp = dataframe[list(columns) + ["category_id"]].copy()
    temp["key"] = build_rule_keys_vendor_code_normalized(temp, columns)
    temp = temp[temp["key"].map(has_meaningful_key)]
    counts = temp.groupby("key")["category_id"].nunique()
    unique_keys = counts[counts == 1].index
    if len(unique_keys) == 0:
        return {}
    unique_rows = temp[temp["key"].isin(unique_keys)].drop_duplicates("key")
    return dict(zip(unique_rows["key"], unique_rows["category_id"]))


def build_ambiguous_category_map(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    max_categories: int,
) -> dict[str, np.ndarray]:
    temp = dataframe[list(columns) + ["category_id"]].copy()
    temp["key"] = build_rule_keys(temp, columns)
    temp = temp[temp["key"].map(has_meaningful_key)]
    if len(temp) == 0:
        return {}

    grouped = temp.groupby("key")["category_id"].unique()
    result: dict[str, np.ndarray] = {}
    for key, categories in grouped.items():
        if categories is None:
            continue
        cats = np.asarray(categories, dtype=np.int32)
        if len(cats) <= 1:
            continue
        if len(cats) > int(max_categories):
            continue
        result[str(key)] = cats
    return result


def build_ambiguous_category_map_with_vendor_code_norm(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    max_categories: int,
) -> dict[str, np.ndarray]:
    temp = dataframe[list(columns) + ["category_id"]].copy()
    temp["key"] = build_rule_keys_vendor_code_normalized(temp, columns)
    temp = temp[temp["key"].map(has_meaningful_key)]
    if len(temp) == 0:
        return {}

    grouped = temp.groupby("key")["category_id"].unique()
    result: dict[str, np.ndarray] = {}
    for key, categories in grouped.items():
        if categories is None:
            continue
        cats = np.asarray(categories, dtype=np.int32)
        if len(cats) <= 1:
            continue
        if len(cats) > int(max_categories):
            continue
        result[str(key)] = cats
    return result


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


def fit_title_feature_blocks(
    texts: list[str],
) -> tuple[TfidfVectorizer, TfidfVectorizer, sparse.csr_matrix]:
    word_vectorizer = TfidfVectorizer(**TITLE_WORD_VECTORIZER_PARAMS)
    char_vectorizer = TfidfVectorizer(**TITLE_CHAR_VECTORIZER_PARAMS)
    word_features = word_vectorizer.fit_transform(texts)
    char_features = char_vectorizer.fit_transform(texts)
    features = sparse.hstack(
        [word_features, char_features],
        format="csr",
        dtype=np.float32,
    )
    return word_vectorizer, char_vectorizer, features


def transform_title_feature_blocks(
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


def build_category_centroids(
    train_matrix: sparse.csr_matrix,
    train_categories: np.ndarray,
    category_department_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix, dict[int, np.ndarray]]:
    category_labels, inverse_indices = np.unique(
        train_categories,
        return_inverse=True,
    )
    row_indices = inverse_indices.astype(np.int32, copy=False)
    col_indices = np.arange(train_matrix.shape[0], dtype=np.int32)
    weights = np.ones(train_matrix.shape[0], dtype=np.float32)

    assignment = sparse.csr_matrix(
        (weights, (row_indices, col_indices)),
        shape=(len(category_labels), train_matrix.shape[0]),
        dtype=np.float32,
    )
    counts = np.asarray(assignment.sum(axis=1)).ravel()
    normalizer = sparse.diags(
        1.0 / np.maximum(counts, 1.0),
        dtype=np.float32,
    )
    category_centroid_matrix = (normalizer @ assignment @ train_matrix).tocsr()
    category_departments = np.array(
        [category_department_map[int(category)] for category in category_labels],
        dtype=np.int16,
    )
    department_category_indices = {
        int(department): np.flatnonzero(category_departments == department).astype(
            np.int32
        )
        for department in np.unique(category_departments)
    }
    return (
        category_labels.astype(np.int32, copy=False),
        category_departments,
        category_centroid_matrix,
        department_category_indices,
    )


def fit_bundle(dataframe: pd.DataFrame) -> dict[str, object]:
    texts = build_search_texts(dataframe)
    title_texts = build_title_texts(dataframe)
    word_vectorizer, char_vectorizer, train_matrix = fit_feature_blocks(texts)
    title_word_vectorizer, title_char_vectorizer, title_matrix = fit_title_feature_blocks(
        title_texts
    )

    department_model = LinearSVC(**DEPARTMENT_MODEL_PARAMS)
    department_model.fit(train_matrix, dataframe["department_id"])

    train_categories = dataframe["category_id"].to_numpy(dtype=np.int32)
    category_department_map = (
        dataframe.drop_duplicates("category_id")
        .set_index("category_id")["department_id"]
        .astype(np.int64)
        .to_dict()
    )
    (
        category_labels,
        category_departments,
        category_centroid_matrix,
        department_category_indices,
    ) = build_category_centroids(
        train_matrix,
        train_categories,
        category_department_map,
    )
    (
        title_category_labels,
        _,
        category_title_centroid_matrix,
        _,
    ) = build_category_centroids(
        title_matrix,
        train_categories,
        category_department_map,
    )
    if not np.array_equal(category_labels, title_category_labels):
        raise ValueError("Category labels are inconsistent across feature views")

    override_maps = []
    for columns in OVERRIDE_RULES:
        override_maps.append({"columns": columns, "mapping": build_unique_category_map(dataframe, columns)})

    normalized_vendor_code_override_maps = []
    if USE_NORMALIZED_VENDOR_CODE_OVERRIDES:
        for columns in [
            ("vendor_code",),
            ("vendor_name", "vendor_code"),
            ("shop_category_name", "vendor_code"),
        ]:
            mapping = build_unique_category_map_with_vendor_code_norm(dataframe, columns)
            if mapping:
                normalized_vendor_code_override_maps.append(
                    {"columns": columns, "mapping": mapping, "key_kind": "vendor_code_norm"}
                )

    ambiguous_override_maps = []
    for rule in AMBIGUOUS_OVERRIDE_RULES:
        columns = tuple(rule["columns"])
        mapping = build_ambiguous_category_map(
            dataframe,
            columns,
            max_categories=int(rule["store_max_categories"]),
        )
        if mapping:
            ambiguous_override_maps.append(
                {
                    "columns": columns,
                    "mapping": mapping,
                    "apply_max_categories": int(rule["apply_max_categories"]),
                    "min_gain_over_base": float(rule["min_gain_over_base"]),
                }
            )
    ambiguous_vendor_code_norm_maps = []
    if USE_NORMALIZED_VENDOR_CODE_OVERRIDES:
        for rule in [
            {
                "columns": ("vendor_code",),
                "store_max_categories": 7,
                "apply_max_categories": 7,
                "min_gain_over_base": 0.0,
            },
            {
                "columns": ("shop_category_name", "vendor_code"),
                "store_max_categories": 10,
                "apply_max_categories": 10,
                "min_gain_over_base": 0.0,
            },
        ]:
            columns = tuple(rule["columns"])
            mapping = build_ambiguous_category_map_with_vendor_code_norm(
                dataframe,
                columns,
                max_categories=int(rule["store_max_categories"]),
            )
            if mapping:
                ambiguous_vendor_code_norm_maps.append(
                    {
                        "columns": columns,
                        "mapping": mapping,
                        "apply_max_categories": int(rule["apply_max_categories"]),
                        "min_gain_over_base": float(rule["min_gain_over_base"]),
                        "key_kind": "vendor_code_norm",
                    }
                )

    return {
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "title_word_vectorizer": title_word_vectorizer,
        "title_char_vectorizer": title_char_vectorizer,
        "department_model": department_model,
        "category_labels": category_labels,
        "category_departments": category_departments,
        "category_centroid_matrix": category_centroid_matrix,
        "category_title_centroid_matrix": category_title_centroid_matrix,
        "department_category_indices": department_category_indices,
        "category_department_map": category_department_map,
        "override_maps": override_maps,
        "normalized_vendor_code_override_maps": normalized_vendor_code_override_maps,
        "ambiguous_override_maps": ambiguous_override_maps,
        "ambiguous_vendor_code_norm_maps": ambiguous_vendor_code_norm_maps,
    }


def compute_similarity_scores(
    query_matrix: sparse.csr_matrix,
    category_centroid_matrix: sparse.csr_matrix,
    query_title_matrix: sparse.csr_matrix,
    category_title_centroid_matrix: sparse.csr_matrix,
) -> sparse.csr_matrix:
    scores = query_matrix @ category_centroid_matrix.T
    title_scores = query_title_matrix @ category_title_centroid_matrix.T
    return scores + TITLE_BONUS_WEIGHT * title_scores


def nearest_centroid_category_indices(
    query_matrix: sparse.csr_matrix,
    category_centroid_matrix: sparse.csr_matrix,
    query_title_matrix: sparse.csr_matrix,
    category_title_centroid_matrix: sparse.csr_matrix,
    department_category_indices: dict[int, np.ndarray],
    top_departments: np.ndarray,
    uncertain_mask: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    predictions = np.empty(query_matrix.shape[0], dtype=np.int32)
    all_indices = np.arange(category_centroid_matrix.shape[0], dtype=np.int32)

    predicted_departments = top_departments[:, 0]
    confident_query_indices = np.flatnonzero(~uncertain_mask)

    for department in np.unique(predicted_departments[confident_query_indices]):
        query_indices = confident_query_indices[
            predicted_departments[confident_query_indices] == department
        ]
        if len(query_indices) == 0:
            continue

        candidate_indices = department_category_indices.get(
            int(department),
            all_indices,
        )
        candidate_matrix = category_centroid_matrix[candidate_indices]
        candidate_title_matrix = category_title_centroid_matrix[candidate_indices]

        for start in range(0, len(query_indices), batch_size):
            stop = min(start + batch_size, len(query_indices))
            batch_query_indices = query_indices[start:stop]
            scores = compute_similarity_scores(
                query_matrix[batch_query_indices],
                candidate_matrix,
                query_title_matrix[batch_query_indices],
                candidate_title_matrix,
            )
            best_local_indices = np.asarray(scores.argmax(axis=1)).ravel()
            predictions[batch_query_indices] = candidate_indices[best_local_indices]

    for query_index in np.flatnonzero(uncertain_mask):
        candidate_parts = []
        for department in top_departments[query_index]:
            candidate_parts.append(
                department_category_indices.get(
                    int(department),
                    np.empty(0, dtype=np.int32),
                )
            )
        candidate_indices = (
            np.concatenate(candidate_parts) if candidate_parts else all_indices
        )
        if len(candidate_indices) == 0:
            candidate_indices = all_indices
        candidate_matrix = category_centroid_matrix[candidate_indices]
        candidate_title_matrix = category_title_centroid_matrix[candidate_indices]
        scores = compute_similarity_scores(
            query_matrix[query_index : query_index + 1],
            candidate_matrix,
            query_title_matrix[query_index : query_index + 1],
            candidate_title_matrix,
        )
        best_local_index = int(np.asarray(scores.argmax(axis=1)).ravel()[0])
        predictions[query_index] = candidate_indices[best_local_index]
    return predictions


def should_sync_department_to_category(
    query_features: sparse.csr_matrix,
    query_title_features: sparse.csr_matrix,
    selected_category_index: int,
    top_department: int,
    model_bundle: dict[str, object],
) -> bool:
    top_department_indices = model_bundle["department_category_indices"].get(
        int(top_department),
        np.empty(0, dtype=np.int32),
    )
    if len(top_department_indices) == 0:
        return False

    selected_scores = compute_similarity_scores(
        query_features,
        model_bundle["category_centroid_matrix"][
            selected_category_index : selected_category_index + 1
        ],
        query_title_features,
        model_bundle["category_title_centroid_matrix"][
            selected_category_index : selected_category_index + 1
        ],
    )
    top_department_scores = compute_similarity_scores(
        query_features,
        model_bundle["category_centroid_matrix"][top_department_indices],
        query_title_features,
        model_bundle["category_title_centroid_matrix"][top_department_indices],
    )
    selected_score = float(np.asarray(selected_scores.toarray()).ravel()[0])
    best_top_department_score = float(
        np.asarray(top_department_scores.max()).ravel()[0]
    )
    return (
        selected_score - best_top_department_score
    ) >= DEPARTMENT_SYNC_SCORE_GAP_THRESHOLD


def apply_category_overrides_with_flags(
    dataframe: pd.DataFrame,
    base_categories: np.ndarray,
    override_maps: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    predictions = base_categories.copy()
    override_used = np.zeros(len(predictions), dtype=bool)
    if len(predictions) == 0:
        return predictions, override_used

    rule_keys = []
    for rule in override_maps:
        columns = rule["columns"]
        if rule.get("key_kind") == "vendor_code_norm":
            rule_keys.append(build_rule_keys_vendor_code_normalized(dataframe, columns))
        else:
            rule_keys.append(build_rule_keys(dataframe, columns))

    for row_index in range(len(predictions)):
        for rule_index, rule in enumerate(override_maps):
            mapping = rule["mapping"]
            key = rule_keys[rule_index][row_index]
            if has_meaningful_key(key) and key in mapping:
                predictions[row_index] = mapping[key]
                override_used[row_index] = True
                break
    return predictions, override_used


def similarity_score_for_category_index(
    query_features: sparse.csr_matrix,
    query_title_features: sparse.csr_matrix,
    category_index: int,
    model_bundle: dict[str, object],
) -> float:
    scores = compute_similarity_scores(
        query_features,
        model_bundle["category_centroid_matrix"][category_index : category_index + 1],
        query_title_features,
        model_bundle["category_title_centroid_matrix"][category_index : category_index + 1],
    )
    return float(np.asarray(scores.toarray()).ravel()[0])


def apply_ambiguous_overrides(
    dataframe: pd.DataFrame,
    base_category_indices: np.ndarray,
    predictions: np.ndarray,
    override_used: np.ndarray,
    model_bundle: dict[str, object],
    query_matrix: sparse.csr_matrix,
    query_title_matrix: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    if len(predictions) == 0:
        return predictions, override_used
    rules = model_bundle.get("ambiguous_override_maps") or []
    if not rules:
        return predictions, override_used

    category_labels = model_bundle["category_labels"]

    for rule in rules:
        columns = rule["columns"]
        mapping = rule["mapping"]
        apply_max_categories = int(rule["apply_max_categories"])
        min_gain_over_base = float(rule["min_gain_over_base"])
        if rule.get("key_kind") == "vendor_code_norm":
            keys = build_rule_keys_vendor_code_normalized(dataframe, columns)
        else:
            keys = build_rule_keys(dataframe, columns)

        for row_index in np.flatnonzero(~override_used):
            key = str(keys[row_index])
            if not has_meaningful_key(key):
                continue
            candidates = mapping.get(key)
            if candidates is None:
                continue
            if len(candidates) > apply_max_categories:
                continue

            candidate_indices = np.searchsorted(category_labels, candidates).astype(
                np.int32, copy=False
            )
            cand_scores = compute_similarity_scores(
                query_matrix[row_index : row_index + 1],
                model_bundle["category_centroid_matrix"][candidate_indices],
                query_title_matrix[row_index : row_index + 1],
                model_bundle["category_title_centroid_matrix"][candidate_indices],
            )
            cand_arr = np.asarray(cand_scores.toarray()).ravel().astype(np.float32, copy=False)
            if cand_arr.size == 0:
                continue
            best_pos = int(cand_arr.argmax())
            best_score = float(cand_arr[best_pos])
            base_score = similarity_score_for_category_index(
                query_matrix[row_index : row_index + 1],
                query_title_matrix[row_index : row_index + 1],
                int(base_category_indices[row_index]),
                model_bundle,
            )
            if (best_score - base_score) < min_gain_over_base:
                continue

            predictions[row_index] = int(category_labels[int(candidate_indices[best_pos])])
            override_used[row_index] = True

    return predictions, override_used


def apply_category_overrides(
    dataframe: pd.DataFrame,
    base_categories: np.ndarray,
    override_maps: list[dict[str, object]],
) -> np.ndarray:
    predictions, _ = apply_category_overrides_with_flags(
        dataframe,
        base_categories,
        override_maps,
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
    title_texts = build_title_texts(dataframe)
    query_matrix = transform_feature_blocks(
        model_bundle["word_vectorizer"],
        model_bundle["char_vectorizer"],
        texts,
    )
    query_title_matrix = transform_title_feature_blocks(
        model_bundle["title_word_vectorizer"],
        model_bundle["title_char_vectorizer"],
        title_texts,
    )

    top_departments, uncertain_mask = get_department_candidates(
        model_bundle["department_model"],
        query_matrix,
    )
    department_predictions = top_departments[:, 0]
    base_category_indices = nearest_centroid_category_indices(
        query_matrix,
        model_bundle["category_centroid_matrix"],
        query_title_matrix,
        model_bundle["category_title_centroid_matrix"],
        model_bundle["department_category_indices"],
        top_departments,
        uncertain_mask,
    )
    base_categories = model_bundle["category_labels"][base_category_indices]
    category_predictions, override_used = apply_category_overrides_with_flags(
        dataframe,
        base_categories,
        model_bundle["override_maps"],
    )
    if USE_NORMALIZED_VENDOR_CODE_OVERRIDES:
        category_predictions, override_used = apply_category_overrides_with_flags(
            dataframe,
            category_predictions,
            model_bundle.get("normalized_vendor_code_override_maps", []),
        )
    category_predictions, override_used = apply_ambiguous_overrides(
        dataframe,
        base_category_indices,
        category_predictions,
        override_used,
        model_bundle,
        query_matrix,
        query_title_matrix,
    )
    if USE_NORMALIZED_VENDOR_CODE_OVERRIDES:
        category_predictions, override_used = apply_ambiguous_overrides(
            dataframe,
            base_category_indices,
            category_predictions,
            override_used,
            {
                **model_bundle,
                "ambiguous_override_maps": model_bundle.get("ambiguous_vendor_code_norm_maps", []),
            },
            query_matrix,
            query_title_matrix,
        )
    final_departments = department_predictions.astype(np.int64, copy=True)
    base_category_departments = model_bundle["category_departments"][
        base_category_indices
    ].astype(np.int64, copy=False)
    category_department_map = model_bundle["category_department_map"]
    sync_candidate_indices = np.flatnonzero(
        (~override_used) & (base_category_departments != final_departments)
    )
    for index in sync_candidate_indices:
        if should_sync_department_to_category(
            query_matrix[index : index + 1],
            query_title_matrix[index : index + 1],
            int(base_category_indices[index]),
            int(department_predictions[index]),
            model_bundle,
        ):
            final_departments[index] = base_category_departments[index]

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
