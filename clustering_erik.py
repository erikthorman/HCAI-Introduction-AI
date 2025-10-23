#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
clustering_erik.py

Läser Markdown-filer från två mappar (Före/Efter), gör TF-IDF, klustrar (KMeans),
tar fram topptermer per kluster, jämför perioder och sparar resultat + figurer.

Output (i --out_dir, default: data/analysis_results):
- docs_with_clusters.csv          (varje dokument med kluster, period, filnamn)
- cluster_top_terms.csv           (toppord per kluster)
- cluster_period_distribution.csv (antal dokument per period i varje kluster)
- tfidf_diff_terms.csv            (ord som ökat/minskat mest: efter - före)
- tsne_clusters.png               (2D-plot av dokumenten färgade per kluster)
- tsne_periods.png                (2D-plot färgade per period)
"""

import argparse
import os
import re
import json
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


# ---------- Grund-liten svensk stopwordslista (så vi slipper extra paket) ----------
SWEDISH_STOPWORDS = {
    "och","det","att","i","en","jag","hon","som","han","på","den","med","var",
    "sig","för","så","till","är","men","ett","om","hade","de","av","icke","mig",
    "du","henne","då","sin","nu","har","inte","hans","honom","skulle","hennes",
    "där","min","man","ej","vid","kunde","något","från","ut","när","efter","upp",
    "vi","dem","vara","vad","över","än","dig","kan","sina","här","ha","mot","alla",
    "under","någon","eller","allt","mycket","sedan","ju","denna","själv","detta",
    "åt","utan","varit","hur","ingen","mitt","ni","bli","blev","oss","din","dessa",
    "några","deras","blir","mina","samma","vilken","er","sådan","vår","blivit",
    "dess","inom","mellan","sådant","varför","varje","vilka","ditt","vem","vilket",
    "sitta","sådana","vart","dina","vars","vårt","våra","ert","era","vilkas"
}
# ----------


def nfc(path_str: str) -> str:
    """Normalisera unicode (NFC) för macOS-paths med å/ä/ö etc."""
    return unicodedata.normalize("NFC", path_str)


def load_texts_from_folder(folder_path: str) -> Tuple[List[str], List[str]]:
    """
    Läser alla .md-filer i given mapp (ej rekursivt),
    returnerar (filnamn, textinnehåll).
    """
    folder = Path(nfc(folder_path))
    if not folder.exists():
        raise FileNotFoundError(f"Mapp finns inte: {folder}")

    texts, files = [], []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() == ".md":
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
            except UnicodeDecodeError:
                content = f.read_text(encoding="latin-1", errors="ignore")
            texts.append(content)
            files.append(f.name)
    return files, texts


def simple_clean(text: str) -> str:
    """
    Enkel städning av markdown/URL/nummer för mer stabil TF-IDF.
    Behåll svenska tecken, ta bort kodblock, länkar och överflödigt brus.
    """
    if not isinstance(text, str):
        return ""
    t = text

    # Ta bort kodblock ```...```
    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)

    # Ta bort inline-code `...`
    t = re.sub(r"`[^`]+`", " ", t)

    # Ta bort URLs
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)

    # Ta bort markdown-headings/listmarkörer
    t = re.sub(r"^#+\s*", " ", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*[-*+]\s+", " ", t, flags=re.MULTILINE)

    # Siffror -> mellanslag
    t = re.sub(r"\d+", " ", t)

    # Specialtecken -> mellanslag
    t = re.sub(r"[^\wåäöÅÄÖ\- ]+", " ", t)

    # Komprimera whitespace
    t = re.sub(r"\s+", " ", t, flags=re.MULTILINE).strip()
    return t


def build_vectorizer(max_features: int, ngram_range: Tuple[int, int]) -> TfidfVectorizer:
    """Skapa TF-IDF vectorizer med svensk stopwords-lista."""
    return TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[\wåäöÅÄÖ\-]{2,}\b",
        stop_words=SWEDISH_STOPWORDS,
        max_features=max_features,
        ngram_range=ngram_range,
        dtype=np.float32
    )


def top_terms_from_centers(centers: np.ndarray, terms: np.ndarray, topk: int = 15) -> List[List[str]]:
    """Hämta topp-ord för varje klustercenter."""
    out = []
    for i in range(centers.shape[0]):
        idx = np.argsort(centers[i])[::-1][:topk]
        out.append(terms[idx].tolist())
    return out


def tsne_plot(X_sparse, labels, title, colors=None, out_path: Path = None, random_state: int = 42):
    """
    Gör en stabil 2D-projektion med SVD (50D) -> t-SNE (2D) och ritar scatter.
    X_sparse: scipy-sparse matrix
    labels: array-lika etiketter (kluster-id eller period)
    """
    # 1) SVD ned till 50D för fart/stabilitet
    svd = TruncatedSVD(n_components=50, random_state=random_state)
    X_50 = svd.fit_transform(X_sparse)

    # 2) t-SNE
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                random_state=random_state, perplexity=30)
    X_2d = tsne.fit_transform(X_50)

    # 3) Plot
    plt.figure(figsize=(10, 7))
    labels = np.array(labels)

    if colors is None:
        # generera enkla färger
        uniq = np.unique(labels)
        cmap = plt.cm.get_cmap("tab10", len(uniq))
        color_map = {lab: cmap(i) for i, lab in enumerate(uniq)}
    else:
        color_map = colors

    for lab in np.unique(labels):
        pts = X_2d[labels == lab]
        plt.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.7, label=str(lab),
                    c=[color_map[lab]])
    plt.title(title)
    plt.legend(loc="best", markerscale=1.2, frameon=False)
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=160)
    plt.close()


def compute_tfidf_diff(X: np.ndarray, terms: np.ndarray, period: pd.Series,
                       topk: int = 40) -> pd.DataFrame:
    """
    Beräknar genomsnittlig TF-IDF per period och skillnad (efter - före).
    Returnerar en DataFrame med topp ökningar och minskningar.
    """
    # Indeces
    mask_after = (period == "efter").values
    mask_before = (period == "före").values

    # Medel per term
    mean_after = X[mask_after].mean(axis=0)
    mean_before = X[mask_before].mean(axis=0)

    # Om X är sparse matrix
    if not isinstance(mean_after, np.ndarray):
        mean_after = np.asarray(mean_after).ravel()
        mean_before = np.asarray(mean_before).ravel()

    diff = mean_after - mean_before  # positivt => vanligare efter
    order = np.argsort(diff)

    top_down = order[:topk]   # mest minskat (negativt)
    top_up = order[::-1][:topk]  # mest ökat (positivt)

    rows = []
    for idx in top_up:
        rows.append({"term": terms[idx], "diff_after_minus_before": float(diff[idx]), "direction": "up_after"})
    for idx in top_down:
        rows.append({"term": terms[idx], "diff_after_minus_before": float(diff[idx]), "direction": "down_after"})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Klustring och ordanvändnings-analys (Före/Efter).")
    parser.add_argument("--before_dir", type=str, default="data/Före Markdown", help="Mapp med markdown före.")
    parser.add_argument("--after_dir", type=str, default="data/Efter Markdown", help="Mapp med markdown efter.")
    parser.add_argument("--out_dir", type=str, default="data/analysis_results", help="Output-katalog.")
    parser.add_argument("--clusters", type=int, default=6, help="Antal kluster (KMeans).")
    parser.add_argument("--max_features", type=int, default=3000, help="Max antal TF-IDF-funktioner.")
    parser.add_argument("--ngrams", nargs=2, type=int, default=[1, 2], help="ngram-range, ex: --ngrams 1 2")
    parser.add_argument("--top_terms", type=int, default=15, help="Antal toppord per kluster.")
    args = parser.parse_args()

    out_dir = Path(nfc(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Ladda data
    print("Läser in markdown...")
    before_files, before_texts = load_texts_from_folder(args.before_dir)
    after_files, after_texts = load_texts_from_folder(args.after_dir)

    if len(before_texts) == 0 and len(after_texts) == 0:
        raise RuntimeError("Hittade inga .md-filer i angivna mappar.")

    df_before = pd.DataFrame({"filename": before_files, "text": before_texts, "period": "före"})
    df_after = pd.DataFrame({"filename": after_files, "text": after_texts, "period": "efter"})
    df = pd.concat([df_before, df_after], ignore_index=True)

    # 2) Rensa text lätt
    print("Städar text...")
    df["clean_text"] = df["text"].map(simple_clean)

    # 3) TF-IDF
    print("TF-IDF beräkning...")
    ngram_range = (int(args.ngrams[0]), int(args.ngrams[1]))
    vectorizer = build_vectorizer(max_features=args.max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(df["clean_text"])
    terms = vectorizer.get_feature_names_out()

    # 4) KMeans-klustring
    print(f"Klustrar dokument (KMeans, k={args.clusters})...")
    kmeans = KMeans(n_clusters=args.clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    df["cluster"] = labels

    # 5) Topptermer per kluster
    print("Tar ut topptermer per kluster...")
    if hasattr(kmeans, "cluster_centers_"):
        centers = kmeans.cluster_centers_
    else:
        # för äldre sklearn-versioner: approximera via medel av medlemmar
        centers = np.zeros((args.clusters, X.shape[1]), dtype=np.float32)
        for k in range(args.clusters):
            members = X[df["cluster"] == k]
            if members.shape[0] > 0:
                centers[k] = members.mean(axis=0).A1

    top_lists = top_terms_from_centers(centers, terms, topk=args.top_terms)
    rows = []
    for k, words in enumerate(top_lists):
        for rank, w in enumerate(words, 1):
            rows.append({"cluster": k, "rank": rank, "term": w})
    df_top = pd.DataFrame(rows)

    # 6) Fördelning per period i varje kluster
    dist = df.groupby(["cluster", "period"]).size().reset_index(name="count")
    dist_pivot = dist.pivot(index="cluster", columns="period", values="count").fillna(0).astype(int).reset_index()

    # 7) TF-IDF-differenser (efter - före)
    print("Beräknar ord som ökat/minskat över tid...")
    df_diff = compute_tfidf_diff(X, terms, df["period"], topk=40)

    # 8) Visualiseringar (t-SNE)
    print("Gör 2D-visualiseringar (t-SNE)...")
    tsne_plot(
        X, labels=df["cluster"].values,
        title="Dokumentkluster (KMeans)",
        out_path=out_dir / "tsne_clusters.png"
    )
    tsne_plot(
        X, labels=df["period"].values,
        title="Dokument per period (före/efter)",
        out_path=out_dir / "tsne_periods.png"
    )

    # 9) Spara resultat
    print("Sparar resultat...")
    df_out = df[["filename", "period", "cluster"]].copy()
    df_out.to_csv(out_dir / "docs_with_clusters.csv", index=False, encoding="utf-8")
    df_top.to_csv(out_dir / "cluster_top_terms.csv", index=False, encoding="utf-8")
    dist_pivot.to_csv(out_dir / "cluster_period_distribution.csv", index=False, encoding="utf-8")
    df_diff.to_csv(out_dir / "tfidf_diff_terms.csv", index=False, encoding="utf-8")

    # Spara även enklare JSON-sammanfattning
    summary = {
        "n_docs": int(df.shape[0]),
        "n_before": int((df["period"] == "före").sum()),
        "n_after": int((df["period"] == "efter").sum()),
        "n_clusters": int(args.clusters),
        "max_features": int(args.max_features),
        "ngram_range": list(ngram_range),
        "outputs": {
            "docs_with_clusters": str(out_dir / "docs_with_clusters.csv"),
            "cluster_top_terms": str(out_dir / "cluster_top_terms.csv"),
            "cluster_period_distribution": str(out_dir / "cluster_period_distribution.csv"),
            "tfidf_diff_terms": str(out_dir / "tfidf_diff_terms.csv"),
            "tsne_clusters_png": str(out_dir / "tsne_clusters.png"),
            "tsne_periods_png": str(out_dir / "tsne_periods.png"),
        }
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n✅ Klart!")
    print(f"- Dokument+kluster: {summary['outputs']['docs_with_clusters']}")
    print(f"- Topptermer/kluster: {summary['outputs']['cluster_top_terms']}")
    print(f"- Fördelning period/kluster: {summary['outputs']['cluster_period_distribution']}")
    print(f"- Ord som ökat/minskat: {summary['outputs']['tfidf_diff_terms']}")
    print(f"- Figurer: tsne_clusters.png, tsne_periods.png i {out_dir}")


if __name__ == "__main__":
    main()