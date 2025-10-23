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

# 👇 NYTT: Plotly för interaktiva figurer
import plotly.express as px

# Stoppord (svenska) inkl. kommuner/landskap + genitiv-s
import nltk, re, unicodedata
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)
SWEDISH_STOPWORDS = set(stopwords.words("swedish"))

# (valfritt) lägg in kommuner/landskap + "s" från din tidigare lista om du vill
# Här visar jag bara hur du kan hooka in egna stoppord:
EXTRA_STOPWORDS = { "ska", "kommer", "kommun", "kommuner"
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
