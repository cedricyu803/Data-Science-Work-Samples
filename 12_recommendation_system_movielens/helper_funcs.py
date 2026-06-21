import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

# ======= Retreival model =======


def recommend_candidates(
    model,
    user_id,
    user_to_idx,
    user_item_matrix,
    idx_to_movie,
    n_candidates: int = 100,
    return_item_id: bool = True,
):
    """
    ALS candidate generation.
    """

    if user_id not in user_to_idx:
        return []

    user_idx = user_to_idx[user_id]

    user_vector = user_item_matrix[user_idx]

    item_idxs, scores = model.recommend(
        userid=user_idx,
        user_items=user_vector,
        N=n_candidates,
        filter_already_liked_items=True,
    )

    output = []

    if return_item_id:
        for i, item_idx in enumerate(item_idxs):
            output.append(
                (
                    idx_to_movie[item_idx],
                    float(scores[i]),
                )
            )
    else:
        for i, item_idx in enumerate(item_idxs):
            output.append(
                (
                    item_idx,
                    float(scores[i]),
                )
            )

    return output


def evaluate_als_model(
    als_model,
    train_als_user_items,
    val_als_user_items,
    user_item_matrix,
    user_to_idx,
    idx_to_movie,
    n_candidates: int = 100,
):

    recalls = []

    for user_id, true_items in val_als_user_items.items():
        if user_id not in train_als_user_items:
            continue

        candidates = recommend_candidates(
            model=als_model,
            user_id=user_id,
            user_to_idx=user_to_idx,
            user_item_matrix=user_item_matrix,
            idx_to_movie=idx_to_movie,
            n_candidates=n_candidates,
        )

        predicted = {movie_id for movie_id, _ in candidates}

        if len(true_items) == 0:
            continue

        recall = len(predicted & set(true_items)) / len(true_items)

        recalls.append(recall)

    als_recall = np.mean(recalls)
    return als_recall


# ======= Features =======

# ======= Movie features =======


def make_movie_features(movies: pd.DataFrame, train_als: pd.DataFrame):
    movies["genres"] = movies["genres"].fillna("")

    all_genres = sorted(
        {
            g
            for row in movies["genres"]
            for g in row.split("|")
            if g != "(no genres listed)"
        }
    )

    for genre in all_genres:
        movies[f"genre_{genre}"] = (
            movies["genres"].str.contains(genre, regex=False).astype(int)
        )

    genre_cols = [c for c in movies.columns if c.startswith("genre_")]

    movie_features = movies[["movieId"] + genre_cols].copy()

    movie_popularity = train_als.groupby("movieId").size().rename("movie_popularity")

    movie_features = movie_features.merge(
        movie_popularity,
        on="movieId",
        how="left",
    )

    movie_features["movie_popularity"] = movie_features["movie_popularity"].fillna(0)
    return movie_features


# ======= User features =======


def make_user_features(train_als: pd.DataFrame):
    user_features = (
        train_als.groupby("userId")
        .agg(
            user_activity=("movieId", "count"),
            user_avg_rating=("rating", "mean"),
        )
        .reset_index()
    )
    return user_features


# ======= Make train and val sets for ranker model =======


def make_ranker_labelled_dataset(
    als_model,
    train_als_user_items: dict,
    validation_user_items: dict,  # train: val_als_user_items. validation: val_ranker_user_items
    user_item_matrix,
    user_to_idx: dict,
    idx_to_movie: dict,
    n_candidates: int = 100,
    user_features: pd.DataFrame = None,
    movie_features: pd.DataFrame = None,
):

    rows = []

    for user_id, future_items in validation_user_items.items():
        if user_id not in train_als_user_items:
            continue

        # use ALS to generate n_candidates items for each user
        candidates = recommend_candidates(
            model=als_model,
            user_id=user_id,
            user_to_idx=user_to_idx,
            user_item_matrix=user_item_matrix,
            idx_to_movie=idx_to_movie,
            n_candidates=n_candidates,
        )

        for rank_pos, (movie_id, als_score) in enumerate(candidates):
            # label=1 if candidate shows up in future items
            label = int(movie_id in future_items)

            rows.append(
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "als_score": als_score,
                    "als_rank": rank_pos,
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)

    if user_features is not None and not user_features.empty:
        df = df.merge(
            user_features,
            on="userId",
            how="left",
        )

    if movie_features is not None and not movie_features.empty:
        df = df.merge(
            movie_features,
            on="movieId",
            how="left",
        )

    return df


# ======= Evaluate ranker model =======


def evaluate_ranker_model(
    ranker_model, val_ranker: pd.DataFrame, feature_cols: list, k: int = 10
):

    X_val = val_ranker[feature_cols]

    val_ranker["pred"] = ranker_model.predict(X_val)

    precisions = []
    recalls = []
    ndcgs = []

    for user_id, group_df in val_ranker.groupby("userId"):
        group_df = group_df.sort_values(
            "pred",
            ascending=False,
        )

        y_true = group_df["label"].values
        y_score = group_df["pred"].values

        positives = y_true.sum()

        if positives == 0:
            continue

        topk = group_df.head(k)

        hits = topk["label"].sum()

        precision = hits / k
        recall = hits / positives

        precisions.append(precision)
        recalls.append(recall)

        ndcg = ndcg_score(
            [y_true],
            [y_score],
            k=k,
        )

        ndcgs.append(ndcg)

    ranker_precision = np.mean(precisions)
    ranker_recall = np.mean(recalls)
    ranker_ndcg = np.mean(ndcgs)

    print("\nFinal metrics")
    print(f"Precision@{k}: {ranker_precision:.4f}")

    print(f"Recall@{k}: {ranker_recall:.4f}")

    print(f"NDCG@{k}: {ranker_ndcg:.4f}")
    return ranker_precision, ranker_recall, ranker_ndcg


# ======= Inference =======


def recommend_for_user(
    user_id: int,
    als_model,
    ranker_model,
    user_item_matrix,
    user_to_idx,
    idx_to_movie,
    feature_cols,
    n_candidates: int = 100,
    k: int = 10,
    user_features: pd.DataFrame = None,
    movie_features: pd.DataFrame = None,
):
    """
    Production-style inference.

    ALS:
        user -> top-N candidates

    Ranker:
        candidate features -> rerank

    Returns
    -------
    pd.DataFrame
        top-k recommendations
    """
    if user_id not in user_to_idx:
        raise ValueError(f"Unknown user_id={user_id}")

    als_recs = recommend_candidates(
        model=als_model,
        user_id=user_id,
        user_to_idx=user_to_idx,
        user_item_matrix=user_item_matrix,
        idx_to_movie=idx_to_movie,
        n_candidates=n_candidates,
        return_item_id=False,  # get item_idx instead
    )

    rows = []

    for rank_pos, (item_idx, als_score) in enumerate(als_recs):
        movie_id = idx_to_movie[item_idx]

        rows.append(
            {
                "userId": user_id,
                "movieId": movie_id,
                "als_score": float(als_score),
                "als_rank": rank_pos,
            }
        )

    candidates = pd.DataFrame(rows)

    if len(candidates) == 0:
        return candidates

    if user_features is not None and not user_features.empty:
        candidates = candidates.merge(
            user_features,
            on="userId",
            how="left",
        )

    if movie_features is not None and not movie_features.empty:
        candidates = candidates.merge(
            movie_features,
            on="movieId",
            how="left",
        )

    candidates[feature_cols] = candidates[feature_cols].fillna(0)

    candidates["ranker_score"] = ranker_model.predict(candidates[feature_cols])

    candidates = candidates.sort_values(
        "ranker_score",
        ascending=False,
    )

    return candidates.head(k)
