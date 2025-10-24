import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Embeddings / Topic modeling
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan
import inspect

# 👇 NYTT: Plotly för interaktiva figurer
import plotly.express as px

# Stoppord (svenska) inkl. kommuner/landskap + genitiv-s
import nltk, re, unicodedata
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)
SWEDISH_STOPWORDS = set(stopwords.words("swedish"))

# (valfritt) lägg in kommuner/landskap + "s" från din tidigare lista om du vill
# Här visar jag bara hur du kan hooka in egna stoppord:
EXTRA_STOPWORDS = { "ska", "kommer", "kommun", "kommuner", "kommunens", "emmaboda", "vännäs", "sätt", "rätt", "genom", "kommunkoncernen", "samt", "image"
    # exempel: "värmland","värmlands","stockholm","stockholms", ...
}
SWEDISH_STOPWORDS.update(EXTRA_STOPWORDS)

# --- Svenska kommuner (SCB-lista, 2024) ---
SWEDISH_MUNICIPALITIES = {
    "alingsås","alvesta","aneby","arboga","arjeplog","arvidsjaur","arvika","askersund",
    "avesta","bengtsfors","berg","bjurholm","bjuv","boden","bollebygd","bollnäs","borgholm",
    "borlänge","borås","botkyrka","boxholm","bromölla","bräcke","burlöv","båstad","dals-ed",
    "danderyd","degervors","dorotea","eda","eksjö","emaboda","enköping","eskilstuna","eslöv",
    "essunga","fagersta","falkenberg","falköping","falun","filipstad","finspång","flen",
    "forshaga","färgelanda","gagnef","gislaved","gnesta","gnosjö","gotland","grums",
    "grästorp","gullspång","gällivare","gävle","göteborg","götene","hagfors","hallsberg",
    "hallstahammar","hammarö","haninge","haparanda","heby","hedemora","helsingborg",
    "herrljunga","hjo","hofors","huddinge","hudiksvall","hultsfred","hylte","håbo",
    "hällefors","härjedalen","härnösand","härryda","hässleholm","höganäs","högsby","hörby",
    "höör","jokkmokk","jönköping","kalix","kalmar","karlsborg","karlshamn","karlskoga",
    "karlskrona","karlstad","katrineholm","kil","kinda","kiruna","klippan","knivsta","kramfors",
    "krokom","kumla","kungsbacka","kungsör","kungälv","kävlinge","köping","laholm","landskrona",
    "laxå","lekeberg","leksand","lerum","lidingö","lidköping","lilla edet","lindesberg",
    "linköping","ljungby","ljusdal","ljusnarsberg","lomma","ludvika","luleå","lund",
    "lycksele","lysekil","malmö","malung-sälen","mariestad","mark","markaryd","mellerud",
    "mjölby","mora","motala","mullsjö","munkedal","munkfors","mölndal","mönsterås","mörbylånga",
    "nacka","nora","norberg","nordanstig","nordmaling","norrköping","norrtälje","nybro","nykvarn",
    "nyköping","nynäshamn","ockelbo","osby","oskarshamn","oxelösund","pajala","partille",
    "perstorp","piteå","ragunda","robertsfors","ronneby","rättvik","sala","salem","sandviken",
    "sigtuna","simrishamn","sjöbo","skara","skellefteå","skinnskatteberg","skurup","skövde",
    "smedjebacken","sollefteå","solna","sorsele","sotenäs","staffanstorp","stenungsund",
    "stockholm","storfors","storuman","strängnäs","strömstad","sundbyberg","sundsvall",
    "sunne","surahammar","svalöv","svedala","svenljunga","säffle","sävsjö","söderhamn",
    "söderköping","södertälje","sölvesborg","tanum","tibro","tidaholm","tierp","timrå",
    "tingsryd","tjörn","tomelilla","torsby","torsås","tranemo","tranås","trelleborg","trollhättan",
    "trosa","tyresö","täby","töreboda","uddevalla","ulricehamn","umeå","upplands väsby","upplands-bro",
    "uppsala","vadstena","vaggeryd","valdemarsvik","vallentuna","vansbro","vara","varberg",
    "vaxholm","vellinge","vetlanda","vilhelmina","vimmerby","vindeln","vingåker","vårgårda",
    "vänersborg","värmdö","värnamo","västervik","västerås","växjö","ydre","ystad","åmål","ånge",
    "åre","årjäng","åsele","åstorp","åtvidaberg","öckerö","ödeshög","örebro","örnsköldsvik",
    "örkelljunga","östersund","östhammar","östra göinge"
}

# --- Svenska landskap ---
SWEDISH_PROVINCES = {
    "blekinge","bohuslän","dalarna","dalsland","gotland","gästrikland","halland","hälsingland",
    "härjedalen","jämtland","lappland","medelpad","norrbotten","närke","skåne","småland",
    "södermanland","uppland","värmland","västerbotten","västergötland","ångermanland","öland","östergötland"
}

# --- Lägg till kommuner, deras genitivformer och landskap till stopporden ---
COMMUNE_STOPWORDS = set()
for name in SWEDISH_MUNICIPALITIES:
    COMMUNE_STOPWORDS.add(name)
    COMMUNE_STOPWORDS.add(name + "s")  # genitivform

PROVINCE_STOPWORDS = set()
for name in SWEDISH_PROVINCES:
    PROVINCE_STOPWORDS.add(name)
    PROVINCE_STOPWORDS.add(name + "s")  # genitivform

SWEDISH_STOPWORDS.update(COMMUNE_STOPWORDS)
SWEDISH_STOPWORDS.update(PROVINCE_STOPWORDS)

# --- Sökvärdar ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEFORE_GLOB = os.path.join(BASE_DIR, "data", "Före Markdown", "*.md")
AFTER_GLOB  = os.path.join(BASE_DIR, "data", "Efter Markdown", "*.md")

def normalize_text(s: str) -> str:
    # Unicode-normalisera och rensa dolda tecken
    return unicodedata.normalize("NFKC", s)

def normalize_text(s: str) -> str:
    # Unicode-normalisera och rensa dolda tecken
    return unicodedata.normalize("NFKC", s)

# --- NYTT: Markdown-aware cleaning + (optional) lemmatization ---
def strip_markdown(text: str) -> str:
    """Remove frontmatter, code blocks, markdown markup and URLs but keep link text."""
    # YAML frontmatter
    text = re.sub(r"^---[\s\S]*?---\s*", " ", text, flags=re.MULTILINE)
    # Fenced code blocks
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Inline code
    text = re.sub(r"`[^`]+`", " ", text)
    # HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Images: ![alt](url) -> alt
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", text)
    # Links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Headings (# ...) -> remove leading hashes
    text = re.sub(r"(^|\n)\s{0,3}#{1,6}\s*", "\n", text)
    # Blockquotes, lists etc at line-start
    text = re.sub(r"(^|\n)\s*[>*+-]\s+", "\n", text)
    # Remove table pipes
    text = text.replace("|", " ")
    # URLs remaining
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return text

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text: str, lemmatize: bool = False) -> str:
    """
    Clean markdown and basic normalization.
    - Keeps words (including åäö) and numbers removed.
    - Does NOT aggressively remove rare tokens so unique 'efter' words remain.
    - Set lemmatize=True to run spaCy Swedish lemmatizer (optional, will try to download if missing).
    """
    t = normalize_text(text)
    t = strip_markdown(t)
    t = t.lower()
    # Keep a-z and åäö, replace other punctuation with space
    t = re.sub(r"[^a-zåäö0-9\s]", " ", t)
    # Remove standalone numbers (optionally keep if you want)
    t = re.sub(r"\b\d+\b", " ", t)
    t = normalize_whitespace(t)

    if lemmatize:
        try:
            import spacy
            try:
                nlp = spacy.load("sv_core_news_sm")
            except Exception:
                # attempt download if model missing
                import subprocess, sys
                print("Downloading spaCy Swedish model (sv_core_news_sm)...")
                subprocess.run([sys.executable, "-m", "spacy", "download", "sv_core_news_sm"], check=True)
                nlp = spacy.load("sv_core_news_sm")
            doc = nlp(t)
            # keep lemmas but avoid removing domain-specific tokens via spaCy stopwords here
            t = " ".join(tok.lemma_ for tok in doc if not tok.is_punct)
            t = normalize_whitespace(t)
        except Exception:
            print("Warning: spaCy lemmatization unavailable — continuing without lemmatization.")

    return t

def load_markdown_texts(path_pattern, period_label, lemmatize=False):
    texts, files, labels = [], [], []
    for filepath in sorted(glob.glob(path_pattern)):
        raw = Path(filepath).read_text(encoding="utf-8")
        raw = normalize_text(raw).strip()
        # hoppa över väldigt korta filer
        if len(raw) < 20:
            continue
        cleaned = preprocess_text(raw, lemmatize=lemmatize)
        # skip if cleaning removed almost everything
        if len(cleaned) < 10:
            continue
        texts.append(cleaned)
        files.append(Path(filepath).name)
        labels.append(period_label)
    return texts, files, labels

def load_markdown_texts(path_pattern, period_label):
    texts, files, labels = [], [], []
    for filepath in sorted(glob.glob(path_pattern)):
        text = Path(filepath).read_text(encoding="utf-8")
        text = normalize_text(text).strip()
        # hoppa över väldigt korta filer
        if len(text) < 20:
            continue
        texts.append(text)
        files.append(Path(filepath).name)
        labels.append(period_label)
    return texts, files, labels

texts_before, files_before, labels_before = load_markdown_texts(BEFORE_GLOB, "före")
texts_after,  files_after,  labels_after  = load_markdown_texts(AFTER_GLOB, "efter")

texts = texts_before + texts_after
files = files_before + files_after
periods = labels_before + labels_after

print(f"Totalt {len(texts)} dokument lästa ({len(texts_before)} före, {len(texts_after)} efter).")

# --- Svenska BERT-embeddings från KB-Lab ---
embed_model_name = "KBLab/sentence-bert-swedish-cased"
embedder = SentenceTransformer(embed_model_name)

print("Beräknar embeddings ...")
embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

# --- Bygg BERTopic med svensk tokenisering/stoppord ---
# CountVectorizer används internt i BERTopic för c-TF-IDF-tematermer
vectorizer = CountVectorizer(
    stop_words=list(SWEDISH_STOPWORDS),
    lowercase=True,
    token_pattern=r"\b[a-zA-ZåäöÅÄÖ]{2,}\b",
    ngram_range=(1, 3)  # tillåt bi-/trigram för bättre temaetiketter
)

# UMAP + HDBSCAN inställningar (kan finjusteras)
umap_model = umap.UMAP(
    n_neighbors=15,
    n_components=5,   # projicera till 5D för klustring (sen gör vi separat 2D för plot)
    min_dist=0.0,
    metric="cosine",
    random_state=42
)
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=3,
    min_samples=1,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

topic_model = BERTopic(
    embedding_model=embedder,     # ger konsekventa embeddings internt
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    language="multilingual",      # explicit språkhantering av tokenisering är via vectorizer ovan
    top_n_words=10,               # hur många toppord som ska beskriva ett tema
    calculate_probabilities=True,
    verbose=True
)

# --- Fit-transform (vi skickar också in förberäknade embeddings för hastighet/stabilitet) ---
topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)

# --- OPTIONAL: reducera/merga mycket lika teman för att minska dubbletter ---
# Experimentera med topic_similarity_threshold (0.25-0.45) eller ange nr_topics
# Detta returnerar en ny model (eller uppdaterar) och vi hämtar nya topic-tilldelningar.
try:
    sig = inspect.signature(topic_model.reduce_topics)
    params = sig.parameters

    if "topic_similarity_threshold" in params:
        # nyare BERTopic-version som stödjer likhets-tröskel
        topic_model = topic_model.reduce_topics(
            docs=texts,
            topics=topics,
            probabilities=probs,
            topic_similarity_threshold=0.35
        )
    elif "nr_topics" in params:
        # äldre/stabil variant: ange önskat antal teman istället
        topic_model = topic_model.reduce_topics(
            docs=texts,
            nr_topics=30
        )
    else:
        # sista utväg: försök utan extra argument
        topic_model = topic_model.reduce_topics(docs=texts)

    # Re-transformera dokument för att få uppdaterade topic-assignments
    topics, probs = topic_model.transform(texts)
    print("Info: topics reduced/merged to reduce duplicates.")
except Exception as e:
    print("Warning: reduce_topics misslyckades:", e)

# --- Skapa 2D-koordinater för visualisering (separat UMAP till 2D för snygg plot) ---
print("Skapar 2D-embedding för visualisering ...")
umap_2d = umap.UMAP(
    n_neighbors=15,
    n_components=2,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)
coords_2d = umap_2d.fit_transform(embeddings)

# --- DataFrame med dokument, period, tema, 2D-koordinater ---
df = pd.DataFrame({
    "file": files,
    "period": periods,
    "topic": topics,
    "x": coords_2d[:, 0],
    "y": coords_2d[:, 1],
})

# Lägg till label per tema (BERTopic genererar etiketter av toppord)
def topic_label(topic_id: int) -> str:
    if topic_id == -1:
        return "T-1: Outlier"
    words = topic_model.get_topic(topic_id)
    if not words:
        return f"T{topic_id}: (tomt)"
    top_terms = ", ".join(w for w, _ in words[:5])
    return f"T{topic_id}: {top_terms}"

df["topic_label"] = df["topic"].apply(topic_label)

# =========================
# 🔵 PLOTLy: INTERAKTIVA GRAFIKER
# =========================

# 1) Interaktiv topics-översikt (kluster/teman med toppord)
fig_topics = topic_model.visualize_topics()
fig_topics.write_html(os.path.join(BASE_DIR, "bertopic_topics.html"))

# 2) Interaktiv dokument-plot i 2D (UMAP) med hover som visar dokument och tema
#    (BERTopic räknar själv en 2D-reducering internt om vi inte ger reduced_embeddings)
fig_docs = topic_model.visualize_documents(
    texts,
    embeddings=embeddings,      # använd dina embeddings för konsekvent projicering
    custom_labels=True          # använd interna namn/toppord i hover
)
fig_docs.write_html(os.path.join(BASE_DIR, "bertopic_documents.html"))

# 3) Extra: egen Plotly-scatter (UMAP 2D vi redan beräknat) färgad per period + hover
fig_period = px.scatter(
    df, x="x", y="y",
    color="period",
    hover_data={"file": True, "topic": True, "topic_label": True, "x": False, "y": False},
    title="UMAP 2D – färgad per period (hover visar fil & tema)"
)
fig_period.write_html(os.path.join(BASE_DIR, "umap_period_scatter.html"))

print("\n🖼️ Plotly-figurer sparade som:")
print(" - bertopic_topics.html")
print(" - bertopic_documents.html")
print(" - umap_period_scatter.html")

# --- (valfritt) Matplotlib-plot finns kvar om du vill ha statisk bild ---
plt.figure(figsize=(10, 8))
for (period), g in df.groupby("period"):
    plt.scatter(g["x"], g["y"], alpha=0.75, s=50, label=period)
plt.title("Tematiska kluster med BERTopic (UMAP 2D)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(title="Period")
plt.tight_layout()
plt.show()

# --- Översikt över teman ---
topic_info = topic_model.get_topic_info()  # kolumner: Topic, Count, Name

def build_name(row):
    tid = int(row["Topic"])
    if tid == -1:
        return "T-1: Outlier"
    words = topic_model.get_topic(tid)
    top_terms = ", ".join(w for w, _ in (words[:5] or []))
    return f"T{tid}: {top_terms}" if top_terms else f"T{tid}"

topic_info["Name"] = topic_info.apply(build_name, axis=1)

# --- Temafördelning före/efter ---
dist = (
    df.groupby(["topic_label", "period"])
      .size()
      .reset_index(name="count")
      .pivot(index="topic_label", columns="period", values="count")
      .fillna(0)
      .astype(int)
)

# Säkerställ kolumnerna även om en period saknas i datat
for col in ["före", "efter"]:
    if col not in dist.columns:
        dist[col] = 0

# Sortera efter total antal dokument per tema (flest överst)
dist["_total"] = dist.sum(axis=1)
dist = dist.sort_values("_total", ascending=False).drop(columns="_total")

# Bygg counts per topic id (inte bara topic_label) så vi kan sortera + visa label
topic_counts = df.groupby(["topic", "period"]).size().unstack(fill_value=0)
# Säkerställ att båda kolumner finns
for c in ["före", "efter"]:
    if c not in topic_counts.columns:
        topic_counts[c] = 0
topic_counts = topic_counts[["före", "efter"]]

# Mappa till mänsklig label
topic_counts = topic_counts.reset_index().rename(columns={"före":"count_before", "efter":"count_after"})
topic_counts["topic_label"] = topic_counts["topic"].map(lambda t: topic_label(int(t)))

# Beräkna andel 'efter' inom temat, ratio och avvikelse från global andel
topic_counts["share_after"] = topic_counts["count_after"] / (topic_counts["count_before"] + topic_counts["count_after"] + 1e-9)
global_after_share = (df["period"] == "efter").mean()
topic_counts["delta_vs_global"] = topic_counts["share_after"] - global_after_share
topic_counts["ratio_after_over_before"] = (topic_counts["count_after"] + 1) / (topic_counts["count_before"] + 1)

# Sortera och spara top förändringar
top_increased = topic_counts.sort_values("ratio_after_over_before", ascending=False).head(20)
top_decreased = topic_counts.sort_values("ratio_after_over_before", ascending=True).head(20)

# Skriv ut kort i konsolen
print("\n— Top temas som blivit mer vanliga EFTER (ratio efter/före) —")
for _, r in top_increased.iterrows():
    print(f"{r['topic_label']:<40} before={int(r['count_before']):3} after={int(r['count_after']):3} ratio={r['ratio_after_over_before']:.2f} Δ={r['delta_vs_global']:.2f}")

print("\n— Top temas som blivit mindre vanliga EFTER —")
for _, r in top_decreased.iterrows():
    print(f"{r['topic_label']:<40} before={int(r['count_before']):3} after={int(r['count_after']):3} ratio={r['ratio_after_over_before']:.2f} Δ={r['delta_vs_global']:.2f}")

# Export: lägg till blad i excel-filen och spara CSV
change_out_csv = os.path.join(BASE_DIR, "topic_change_summary.csv")
topic_counts.to_csv(change_out_csv, index=False, encoding="utf-8")

# Lägg till som nytt blad i befintlig excel (append/replace)
try:
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        topic_counts.to_excel(writer, sheet_name="topic_change", index=False)
except Exception:
    # fallback: skriv separat workbook om append inte stöds
    topic_counts.to_excel(os.path.join(BASE_DIR, "topic_change_summary.xlsx"), index=False)

print(f"\n💾 Topic change summary saved: {change_out_csv}")
# --- Exportera till Excel med flera flikar ---
out_path = os.path.join(BASE_DIR, "bertopic_results.xlsx")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    topic_info.to_excel(writer, sheet_name="topics_overview", index=False)
    df[["file", "period", "topic", "topic_label"]].to_excel(writer, sheet_name="doc_topics", index=False)
    dist.to_excel(writer, sheet_name="topic_period_distribution")

print(f"\n💾 Resultat sparade till: {out_path}")
print("Flikar: topics_overview, doc_topics, topic_period_distribution")

# --- (valfritt) Skriv ut toppord per tema i konsolen ---
print("\n— Teman och toppord —")
for tid in sorted(df["topic"].unique()):
    print(topic_label(tid))

import math
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2

def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction (returns array of q-values)."""
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n+1)
    q = p * n / ranks
    # enforce monotonicity
    q_sorted = np.minimum.accumulate(q[np.argsort(-ranks)])[np.argsort(-ranks)]
    return np.minimum(q_sorted, 1.0)

def word_change_analysis(texts, periods, stopwords=None, ngram=(1,1), n_top=25, out_dir=BASE_DIR):
    """
    Word-level comparison BEFORE vs AFTER.
    - texts: list of cleaned strings (you already have `texts`)
    - periods: parallel list of "före"/"efter" (you already have `periods`)
    Outputs CSV and plots in out_dir.
    """
    # Build simple CountVectorizer for unigrams (change ngram if desired)
    vec = CountVectorizer(
        stop_words=list(stopwords) if stopwords is not None else None,
        token_pattern=r"\b[a-zA-ZåäöÅÄÖ]{2,}\b",
        lowercase=True,
        ngram_range=ngram
    )
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()

    # binary label: 0 = före, 1 = efter
    y = np.array([1 if p == "efter" else 0 for p in periods])

    # counts per term by group
    counts_before = np.asarray(X[y == 0].sum(axis=0)).ravel()
    counts_after  = np.asarray(X[y == 1].sum(axis=0)).ravel()
    total_before = counts_before.sum()
    total_after  = counts_after.sum()

    # normalized frequencies
    freq_before = counts_before / (total_before + 1e-12)
    freq_after  = counts_after / (total_after + 1e-12)

    # Smoothed log-odds ratio (additive prior alpha)
    alpha = 0.01
    V = len(terms)
    log_odds = np.log((counts_after + alpha) / (total_after + alpha * V)) - np.log((counts_before + alpha) / (total_before + alpha * V))

    # chi2 and p-values (fast vectorized)
    chi2_stats, pvals = chi2(X, y)
    qvals = fdr_bh(pvals)

    # TF-IDF mean difference (after - before)
    tfidf_transformer = TfidfTransformer(norm=None, use_idf=True)
    X_tfidf = tfidf_transformer.fit_transform(X)
    mean_tfidf_before = np.asarray(X_tfidf[y == 0].mean(axis=0)).ravel()
    mean_tfidf_after  = np.asarray(X_tfidf[y == 1].mean(axis=0)).ravel()
    tfidf_diff = mean_tfidf_after - mean_tfidf_before

    # assemble dataframe
    df_terms = pd.DataFrame({
        "term": terms,
        "count_before": counts_before,
        "count_after": counts_after,
        "freq_before": freq_before,
        "freq_after": freq_after,
        "log_odds": log_odds,
        "chi2": chi2_stats,
        "pvalue": pvals,
        "qvalue": qvals,
        "tfidf_diff": tfidf_diff
    })

    # derived measures
    df_terms["support"] = df_terms["count_before"] + df_terms["count_after"]
    df_terms = df_terms.sort_values(by=["log_odds"], ascending=False).reset_index(drop=True)

    # Save CSV
    out_csv = os.path.join(out_dir, "word_change_summary.csv")
    df_terms.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n💾 Word change CSV saved: {out_csv}")

    # Print top increased / decreased words
    print("\nTop words enriched AFTER (log-odds high):")
    for _, r in df_terms[df_terms["support"] >= 3].head(n_top).iterrows():
        print(f"{r['term']:<20} after={int(r['count_after']):3} before={int(r['count_before']):3} log_odds={r['log_odds']:.2f} q={r['qvalue']:.3g}")

    print("\nTop words enriched BEFORE (log-odds low):")
    for _, r in df_terms[df_terms["support"] >= 3].sort_values("log_odds").head(n_top).iterrows():
        print(f"{r['term']:<20} after={int(r['count_after']):3} before={int(r['count_before']):3} log_odds={r['log_odds']:.2f} q={r['qvalue']:.3g}")

    # Simple barplots for top N (increase + decrease)
    try:
        top_pos = df_terms[df_terms["support"] >= 3].head(n_top)
        top_neg = df_terms[df_terms["support"] >= 3].sort_values("log_odds").head(n_top)

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].barh(top_pos["term"][::-1], top_pos["log_odds"][::-1], color="tab:green")
        ax[0].set_title("Top words ↑ AFTER (log-odds)")
        ax[1].barh(top_neg["term"][::-1], top_neg["log_odds"][::-1], color="tab:red")
        ax[1].set_title("Top words ↑ BEFORE (log-odds)")
        plt.tight_layout()
        plot_path = os.path.join(out_dir, "word_change_barplots.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"💾 Barplots saved: {plot_path}")
    except Exception as e:
        print("Warning: could not create barplots:", e)

    # Optional: try to create a wordcloud for AFTER-only top terms (if wordcloud installed)
    try:
        from wordcloud import WordCloud
        after_only = df_terms.sort_values("log_odds", ascending=False).head(100)
        freqs = dict(zip(after_only["term"], (after_only["count_after"]+1)))
        wc = WordCloud(width=1200, height=600, background_color="white", collocations=False, prefer_horizontal=0.9)
        wc.generate_from_frequencies(freqs)
        wc_path = os.path.join(out_dir, "wordcloud_after.png")
        wc.to_file(wc_path)
        print(f"💾 Wordcloud (AFTER) saved: {wc_path}")
    except Exception:
        # silently skip if library missing
        pass

    return df_terms

# Run the analysis using your existing texts & periods and stopwords
print("\nKör ord-analys (före/efter) — detta sparar CSV och plottar top words.")
df_word_changes = word_change_analysis(texts, periods, stopwords=SWEDISH_STOPWORDS, ngram=(1,1), n_top=30)