#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Stoppord (svenska) inkl. kommuner/landskap + genitiv-s
import nltk, unicodedata
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)

# Minska varning från HuggingFace tokenizers vid multiprocess
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


########################################
# Svenska stopwords (manuell lista + NLTK)
########################################

# Manuella extra-stopplistor (behålls och unioneras med NLTK)
MANUAL_EXTRA_STOPWORDS = {
    "och", "men", "det", "att", "som", "inte", "jag", "du", "han", "hon", "den",
    "detta", "de", "vi", "ni", "mig", "dig", "honom", "henne", "oss",
    "er", "min", "mitt", "mina", "din", "ditt", "dina", "sin", "sina", "vår",
    "vårt", "våra", "ert", "era",
    "är", "var", "vara", "varit", "blir", "blev", "bli",
    "har", "hade", "ha",
    "kan", "kunde", "ska", "skulle", "måste", "får", "få",
    "på", "i", "från", "till", "för", "med", "utan", "över", "under",
    "efter", "innan", "mellan", "inom", "utom", "vid", "hos", "om",
    "så", "då", "när", "nu", "här", "där", "dessa", "denna", "detta",
    "vad", "vilken", "vilka", "vilket", "varför", "hur", "varje",
    "också", "även", "mycket", "sådan", "sådana", "samma", "all", "alla",
    "ingen", "inget", "någon", "något", "några",
    "man", "en", "ett", "av",

    # SWEDISH_MUNICIPALITIES
    "ale", "alingsås", "alvesta", "aneby", "arboga", "arjeplogs", "arvidsjaurs", "arvika", "askersunds", "avesta", 
    "bengtsfors", "bergs", "bjurholms", "bjuvs", "bodens", "bollebygds", "bollnäs", "borgholms", "borlänge", "borås", 
    "botkyrka", "boxholms", "bromölla", "bräcke", "burlövs", "båstads", "dals-eds", "danderyds", "degerfors", "dorotea", 
    "eda", "ekerö", "eksjö", "emmaboda", "enköpings", "eskilstuna", "eslövs", "essunga", "fagersta", "falkenbergs", 
    "falköpings", "falu", "filipstads", "finspångs", "flens", "forshaga", "färgelanda", "gagnefs", "gislaveds", "gnesta", 
    "gnosjö", "gotlands", "grums", "grästorps", "gullspångs", "gällivare", "gävle", "göteborgs", "götene", "habo", "hagfors", 
    "hallsbergs", "hallstahammars", "halmstads", "hammarö", "haninge", "haparanda", "heby", "hedemora", "helsingborgs", 
    "herrljunga", "hjo", "hofors", "huddinge", "hudiksvalls", "hultsfreds", "hylte", "håbo", "hällefors", "härjedalens", 
    "härnösands", "härryda", "hässleholms", "höganäs", "högsby", "hörby", "höörs", "jokkmokks", "järfälla", "jönköpings", 
    "kalix", "kalmar", "karlsborgs", "karlshamns", "karlskoga", "karlskrona", "karlstads", "katrineholms", "kils", "kinda", 
    "kiruna", "klippans", "knivsta", "kramfors", "kristianstads", "kristinehamns", "krokoms", "kumla", "kungsbacka", "kungsörs", 
    "kungälvs", "kävlinge", "köpings", "laholms", "landskrona", "laxå", "lekebergs", "leksands", "lerums", "lessebo", "lidingö", 
    "lidköpings", "lilla edets", "lindesbergs", "linköpings", "ljungby", "ljusdals", "ljusnarsbergs", "lomma", "ludvika", "luleå", 
    "lunds", "lycksele", "lysekils", "malmö", "malung-sälens", "malå", "mariestads", "markaryds", "marks", "melleruds", "mjölby", 
    "mora", "motala", "mullsjö", "munkedals", "munkfors", "mölndals", "mönsterås", "mörbylånga", "nacka", "nora", "norbergs", "nordanstigs", 
    "nordmalings", "norrköpings", "norrtälje", "norsjö", "nybro", "nykvarns", "nyköpings", "nynäshamns", "nässjö", "ockelbo", "olofströms", 
    "orsa", "orusts", "osby", "oskarshamns", "ovanåkers", "oxelösunds", "pajala", "partille", "perstorps", "piteå", "ragunda", "robertsfors", 
    "ronneby", "rättviks", "sala", "salems", "sandvikens", "sigtuna", "simrishamns", "sjöbo", "skara", "skellefteå", "skinnskattebergs", 
    "skurups", "skövde", "smedjebackens", "sollefteå", "sollentuna", "solna", "sorsele", "sotenäs", "staffanstorps", "stenungsunds", 
    "stockholms", "storfors", "storumans", "strängnäs", "strömstads", "strömsunds", "sundbybergs", "sundsvalls", "sunne", "surahammars", 
    "svalövs", "svedala", "svenljunga", "säffle", "säters", "sävsjö", "söderhamns", "söderköpings", "södertälje", "sölvesborgs", "tanums", 
    "tibro", "tidaholms", "tierps", "timrå", "tingsryds", "tjörns", "tomelilla", "torsby", "torsås", "tranemo", "tranås", "trelleborgs", 
    "trollhättans", "trosa", "tyresö", "täby", "töreboda", "uddevalla", "ulricehamns", "umeå", "upplands väsby", "upplands-bro", 
    "uppsala", "uppvidinge", "vadstena", "vaggeryds", "valdemarsviks", "vallentuna", "vansbro", "vara", "varbergs", "vaxholms", 
    "vellinge", "vetlanda", "vilhelmina", "vimmerby", "vindelns", "vingåkers", "vårgårda", "vänersborgs", "vännäs", "värmdö", 
    "värnamo", "västerviks", "västerås", "växjö", "ydre", "ystads", "åmåls", "ånge", "åre", "årjängs", "åsele", "åstorps", 
    "åtvidabergs", "älmhults", "älvdalens", "älvkarleby", "älvsbyns", "ängelholms", "öckerö", "ödeshögs", "örebro", "örkelljunga", 
    "örnsköldsviks", "östersunds", "österåkers", "östhammars", "östra göinge", "överkalix", "övertorneå",
    # vanliga tilläggsord i dokument
    "kommun", "kommunen", "kommunal", "kommunstyrelse", "kommunfullmäktige",
    "region", "regionen", "landsting", "stadsdelsförvaltning", "förvaltning",
    "myndighet", "myndigheten", "verksamhet", "organisation",

    #manuella stopwords
    "varav","toverud","utveckl","bengtsfa","gunn","guid","stahl","pris","ocksa","hööra","tjör","unikomfamilj",
    "svalöva","degerfa","årjängas","norrbott", "olofströms","gislaveda","nykvar","emmabod","uppsal","åstorpa",
    "gölisk","svedal","älmhult","itd","munkfor","munkfa","sydnärke","kungsöra","sandvik","årjänga","österåkers",
    "ska", "stockholmarna", "kristianstads","karlstads","kommer", "kommun", "kommuner", "kommunens", "emmaboda", 
    "vännäs", "sätt", "rätt", "genom", "kommunkoncernen", "samt", "image" ,"kr", "nok","pa","mom","ekerö","älmhults",
    "lsa","göliska","eblomlådan","stockholmarnas","sydnärkes","säby", "rönninge","norsjö","degerfors","säby","torg"
    

}

# NLTK:s svenska stoppord
NLTK_SWEDISH = set(stopwords.words("swedish"))

# Sammanfoga NLTK + manuella
SWEDISH_STOPWORDS = set(NLTK_SWEDISH) | set(MANUAL_EXTRA_STOPWORDS)


########################################
# Hjälpfunktioner
########################################

def read_markdown_dir(dir_path, period_label):
    """Läs alla .md-filer rekursivt"""
    docs = []
    for path in glob.glob(os.path.join(dir_path, "**", "*.md"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            docs.append({"doc_id": os.path.basename(path), "period": period_label, "text": text})
        except Exception as e:
            print(f"⚠️ Misslyckades läsa {path}: {e}")
    return docs


def clean_text(text):
    """Rensa text: små bokstäver, ta bort siffror, specialtecken"""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zåäöéüøæ\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_swe_stopwords():
    cleaned = set()
    for w in SWEDISH_STOPWORDS:
        w_clean = clean_text(w)
        for part in w_clean.split():
            if len(part) > 1:
                cleaned.add(part)
    return cleaned


def embed_documents(texts):
    """Generera embeddings med svensk BERT"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")
    return model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)


def run_kmeans(embeddings, k=6):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return kmeans, labels


def top_terms_for_cluster(texts, stop_words, top_n=8):
    """Hitta typiska ord för varje kluster.

    - Säkerställ att stop_words är i ett format som scikit-learn accepterar (lista/None/str).
    - Faller tillbaka till enkel frekvensräkning om TF‑IDF inte går att beräkna
      (t.ex. tomt vokabulär).
    """
    # Säkerställ lista av strängar utan tomma
    if hasattr(texts, "tolist"):
        texts = texts.tolist()
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []

    # SkLearn kräver list/None/str, inte set/tuple
    if isinstance(stop_words, (set, tuple)):
        stop_words_sklearn = sorted(list(stop_words))
    else:
        stop_words_sklearn = stop_words

    try:
        vec = TfidfVectorizer(
            stop_words=stop_words_sklearn, max_features=3000, ngram_range=(1, 2)
        )
        X = vec.fit_transform(texts)
        if X.shape[1] == 0:
            return []
        mean_scores = np.asarray(X.mean(axis=0)).ravel()
        vocab = vec.get_feature_names_out()
        order = np.argsort(mean_scores)[::-1][:top_n]
        return [vocab[i] for i in order]
    except Exception:
        # Fallback: enkel ordfrekvens utan stopwords
        stop_set = set(stop_words_sklearn) if stop_words_sklearn else set()
        c = Counter()
        for t in texts:
            for w in t.split():
                if w not in stop_set and len(w) > 2:
                    c[w] += 1
        return [w for w, _ in c.most_common(top_n)]


def summarize_clusters(df, labels, stop_words):
    """Sammanfatta kluster och deras förändring mellan före/efter"""
    df["cluster"] = labels
    n_before = (df["period"] == "before").sum()
    n_after = (df["period"] == "after").sum()

    summaries = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        sub_before = sub[sub["period"] == "before"]
        sub_after = sub[sub["period"] == "after"]

        share_before = len(sub_before) / n_before if n_before > 0 else 0
        share_after = len(sub_after) / n_after if n_after > 0 else 0
        change = share_after - share_before

        top_terms = top_terms_for_cluster(sub["clean_text"], stop_words)
        summaries.append({
            "cluster": c,
            "top_terms": ", ".join(top_terms),
            "share_before": share_before,
            "share_after": share_after,
            "share_change_after_minus_before": change
        })

    return df, pd.DataFrame(summaries)


def pca_scatter_plot(embeddings, df, summaries, out_path):
    """Visualisera dokumentkluster med temaord"""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    df_plot = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": df["cluster"]
    })

    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap("tab10", len(summaries))

    for i, row in summaries.iterrows():
        sub = df_plot[df_plot["cluster"] == row["cluster"]]
        plt.scatter(sub["x"], sub["y"], color=colors(i), label=f"Kluster {row['cluster']}")
        plt.text(sub["x"].mean(), sub["y"].mean(),
                 f"{row['cluster']}: {row['top_terms'].split(',')[0]}",
                 fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="black", boxstyle="round,pad=0.3"))

    plt.title("Dokumentkluster och temaord (PCA på svenska embeddings)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def compute_word_shift(df, stop_words, out_csv_path, top_n=100):
    """Beräkna ord som ökat mest efter 2023"""
    before_texts = df[df["period"] == "before"]["clean_text"].tolist()
    after_texts = df[df["period"] == "after"]["clean_text"].tolist()

    def count_words(texts):
        c = Counter()
        for t in texts:
            for w in t.split():
                if w not in stop_words and len(w) > 2:
                    c[w] += 1
        return c

    before_counts = count_words(before_texts)
    after_counts = count_words(after_texts)

    total_before = sum(before_counts.values()) or 1
    total_after = sum(after_counts.values()) or 1

    before_freq = {w: before_counts[w] / total_before for w in before_counts}
    after_freq = {w: after_counts[w] / total_after for w in after_counts}

    all_words = set(before_counts.keys()) | set(after_counts.keys())
    data = []
    for w in all_words:
        fb, fa = before_freq.get(w, 0.0), after_freq.get(w, 0.0)
        diff = fa - fb
        data.append({"word": w, "freq_before": fb, "freq_after": fa, "increase": diff})

    df_shift = pd.DataFrame(sorted(data, key=lambda x: x["increase"], reverse=True)[:top_n])
    df_shift.to_csv(out_csv_path, index=False, encoding="utf-8")


########################################
# main()
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before_dir", required=True)
    parser.add_argument("--after_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--clusters", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    stop_words = get_swe_stopwords()

    print("1) Läser och städar text ...")
    docs_before = read_markdown_dir(args.before_dir, "before")
    docs_after = read_markdown_dir(args.after_dir, "after")
    df = pd.DataFrame(docs_before + docs_after)
    df["clean_text"] = df["text"].apply(clean_text)
    print(f"   Totalt {len(df)} dokument")

    print("2) Bygger embeddings ...")
    embeddings = embed_documents(df["clean_text"].tolist())

    print("3) Klustrar dokument ...")
    kmeans, labels = run_kmeans(embeddings, args.clusters)

    print("4) Summerar kluster ...")
    df, summaries = summarize_clusters(df, labels, stop_words)
    cluster_shift_path = os.path.join(args.out_dir, "cluster_shift.csv")
    summaries.to_csv(cluster_shift_path, index=False, encoding="utf-8")

    print("5) Visualiserar kluster ...")
    scatter_path = os.path.join(args.out_dir, "cluster_scatter.png")
    pca_scatter_plot(embeddings, df, summaries, scatter_path)

    print("6) Räknar ord som ökat efter 2023 ...")
    word_shift_path = os.path.join(args.out_dir, "word_shift.csv")
    compute_word_shift(df, stop_words, word_shift_path)

    print("\n✅ Klart! Resultat:")
    print(f"- {scatter_path}")
    print(f"- {cluster_shift_path}")
    print(f"- {word_shift_path}")


if __name__ == "__main__":
    main()
