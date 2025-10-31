import os
import glob
from pathlib import Path
import re
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2

# Stopwords (svenska) inkcluding names for municipalities and provinces and their genitive forms
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
SWEDISH_STOPWORDS = set(stopwords.words("swedish"))

EXTRA_STOPWORDS = {
    "varav","toverud","utveckl","bengtsfa","gunn","guid","stahl","pris","ocksa","hööra","tjör",
    "unikomfamilj","svalöva","degerfa","årjängas","norrbott","olofströms","gislaveda","nykvar",
    "emmabod","uppsal","åstorpa","gölisk","svedal","älmhult","itd","munkfor","munkfa","sydnärke",
    "kungsöra","sandvik","årjänga","österåkers","ska","stockholmarna","kristianstads","karlstads",
    "kommer","kommun","kommuner","kommunens","emmaboda","vännäs","sätt","rätt","genom","kommunkoncernen",
    "samt","image","kr","nok","pa","mom","ekerö","älmhults","lsa","göliska","eblomlådan","stockholmarnas",
    "sydnärkes","säby","rönninge","norsjö","degerfors","säby","torg"
}
SWEDISH_STOPWORDS.update(EXTRA_STOPWORDS)

# --- Swedish municipalities (static list)
SWEDISH_MUNICIPALITIES = {
    "ale","alingsås","alvesta","aneby","arboga","arjeplog","arvidsjaur","arvika","askersund",
    "avesta","bengtsfors","berg","bjurholm","bjuv","boden","bollebygd","bollnäs","borgholm",
    "borlänge","borås","botkyrka","boxholm","bromölla","bräcke","burlöv","båstad","dals-ed",
    "danderyd","degerfors","degervar","dorotea","eda","eksjö","emmaboda","enköping","eskilstuna","eslöv",
    "essunga","fagersta","falkenberg","falköping","falun","filipstad","finspång","flen","forshaga",
    "färgelanda","gagnef","gislaved","gnesta","gnosjö","gotland","grums","grästorp","gullspång",
    "gällivare","gävle","göteborg","götene","hagfors","hallsberg","hallstahammar","hammarö","haninge",
    "haparanda","heby","hedemora","helsingborg","herrljunga","hjo","hofors","huddinge","hudiksvall",
    "hultsfred","hylte","håbo","hällefors","härjedalen","härnösand","härryda","hässleholm","höganäs",
    "högsby","hörby","höör","jokkmokk","järfälla","jönköping","kalix","kalmar","karlsborg","karlshamn",
    "karlskoga","karlskrona","karlstad","katrineholm","kil","kinda","kiruna","klippan","knivsta",
    "kramfors","krokom","kumla","kungsbacka","kungsör","kungälv","kävlinge","köping","laholm",
    "landskrona","laxå","lekeberg","leksand","lerum","lidingö","lidköping","lilla edet","lindesberg",
    "linköping","ljungby","ljusdal","ljusnarsberg","lomma","ludvika","luleå","lund","lycksele","lysekil",
    "malå","malmö","malung-sälen","mariestad","mark","markaryd","mellerud","mjölby","mora","motala",
    "mullsjö","munkedal","munkfors","mölndal","mönsterås","mörbylånga","nacka","nora","norberg",
    "nordanstig","nordmaling","norrköping","norrtälje","nybro","nykvarn","nyköping","nynäshamn","nässjö",
    "ockelbo","olofström","orsa","orust","osby","oskarshamn","oxelösund","pajala","partille","perstorp",
    "piteå","ragunda","robertsfors","ronneby","rättvik","sala","salem","sandviken","sigtuna","simrishamn",
    "sjöbo","skara","skellefteå","skinnskatteberg","skurup","skövde","smedjebacken","sollefteå","solna",
    "sorsele","sotenäs","staffanstorp","stenungsund","stockholm","storfors","storuman","strängnäs","strömstad",
    "sundbyberg","sundsvall","sunne","surahammar","svalöv","svedala","svenljunga","säffle","sävsjö","söderhamn",
    "söderköping","södertälje","sölvesborg","tanum","tibro","tidaholm","tierp","timrå","tingsryd","tjörn",
    "tomelilla","torsby","torsås","tranemo","tranås","trelleborg","trollhättan","trosa","tyresö","täby",
    "töreboda","uddevalla","ulricehamn","umeå","upplands väsby","upplands-bro","uppsala","vadstena","vaggeryd",
    "valdemarsvik","vallentuna","vansbro","vara","varberg","vaxholm","vellinge","vetlanda","vilhelmina","vimmerby",
    "vindeln","vingåker","vårgårda","vänersborg","värmdö","värnamo","västervik","västerås","växjö","ydre",
    "ystad","åmål","ånge","åre","årjäng","åsele","åstorp","åtvidaberg","öckerö","ödeshög","örebro","örnsköldsvik",
    "örkelljunga","östersund","östhammar","östra göinge","överkalix","överums","älvdalen","älvkarleby","åsedal",
    "ås"
}
# --- Swedish provinces (static list)
SWEDISH_PROVINCES = {
    "blekinge","bohuslän","dalarna","dalsland","gotland","gästrikland","halland","hälsingland",
    "härjedalen","jämtland","lappland","medelpad","norrbotten","närke","skåne","småland",
    "södermanland","uppland","värmland","västerbotten","västergötland","ångermanland","öland","östergötland"
}

# --- Adding municipalities and provinces to stopwords (including genitive forms) ---
COMMUNE_STOPWORDS = set()
for name in SWEDISH_MUNICIPALITIES:
    COMMUNE_STOPWORDS.add(name)
    COMMUNE_STOPWORDS.add(name + "s")

PROVINCE_STOPWORDS = set()
for name in SWEDISH_PROVINCES:
    PROVINCE_STOPWORDS.add(name)
    PROVINCE_STOPWORDS.add(name + "s")

SWEDISH_STOPWORDS.update(COMMUNE_STOPWORDS)
SWEDISH_STOPWORDS.update(PROVINCE_STOPWORDS)

# --- Search paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEFORE_GLOB = os.path.join(BASE_DIR, "data", "Före Markdown", "*.md")
AFTER_GLOB  = os.path.join(BASE_DIR, "data", "Efter Markdown", "*.md")


def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

# Markdown-aware cleaning + (optional) lemmatization
def strip_markdown(text: str) -> str:
    """Remove frontmatter, code blocks, markdown markup and URLs but keep link text."""
    text = re.sub(r"^---[\s\S]*?---\s*", " ", text, flags=re.MULTILINE)
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"(^|\n)\s{0,3}#{1,6}\s*", "\n", text)
    text = re.sub(r"(^|\n)\s*[>*+-]\s+", "\n", text)
    text = text.replace("|", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return text


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str, lemmatize: bool = True) -> str:
    """
    Clean markdown and basic normalization.
    - Keeps words (including åäö) and removes standalone numbers.
    - Optionally lemmatizes with spaCy sv_core_news_sm.
    """
    t = normalize_text(text)
    t = strip_markdown(t)
    t = t.lower()
    t = re.sub(r"[^a-zåäö0-9\s]", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)
    t = normalize_whitespace(t)

    if lemmatize:
        try:
            import spacy
            try:
                nlp = spacy.load("sv_core_news_sm")
            except Exception:
                import subprocess, sys
                print("Downloading spaCy Swedish model (sv_core_news_sm)...")
                subprocess.run([sys.executable, "-m", "spacy", "download", "sv_core_news_sm"], check=True)
                nlp = spacy.load("sv_core_news_sm")
            doc = nlp(t)
            t = " ".join(tok.lemma_ for tok in doc if not tok.is_punct)
            t = normalize_whitespace(t)
        except Exception:
            print("Warning: spaCy lemmatization unavailable — continuing without lemmatization.")

    return t

# Load texts from markdown files
def load_markdown_texts(path_pattern, period_label, lemmatize=True):
    texts, files, labels = [], [], []
    for filepath in sorted(glob.glob(path_pattern)):
        raw = Path(filepath).read_text(encoding="utf-8")
        raw = normalize_text(raw).strip()
        if len(raw) < 20:
            continue
        cleaned = preprocess_text(raw, lemmatize=lemmatize)
        if len(cleaned) < 10:
            continue
        texts.append(cleaned)
        files.append(Path(filepath).name)
        labels.append(period_label)
    return texts, files, labels


texts_before, files_before, labels_before = load_markdown_texts(BEFORE_GLOB, "före", lemmatize=True)
texts_after,  files_after,  labels_after  = load_markdown_texts(AFTER_GLOB, "efter", lemmatize=True)

texts = texts_before + texts_after
files = files_before + files_after
periods = labels_before + labels_after

print(f"In total {len(texts)} documents read, ({len(texts_before)} before, {len(texts_after)} after).")


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction (returns q-values)."""
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    ranked = np.empty(n, dtype=int)
    ranked[order] = np.arange(1, n + 1)
    q = p * n / ranked
    # ensure monotonicity
    q_sorted = np.minimum.accumulate(q[::-1])[::-1]
    q_final = np.minimum(q_sorted, 1.0)
    return q_final


def word_change_analysis(texts, periods, stopwords=SWEDISH_STOPWORDS,
                         ngram=(1, 1), n_top=25, plot_n=None, out_dir=BASE_DIR):
    """
    Word-level comparison BEFORE vs AFTER using document-level prevalence
    to correct for different batch sizes.
    """
    vec = CountVectorizer(
        stop_words=list(stopwords) if stopwords is not None else None,
        token_pattern=r"\b[a-zåäöA-ZÅÄÖ]{2,}\b",
        lowercase=True,
        ngram_range=ngram
    )
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()

    y = np.array([1 if p == "efter" else 0 for p in periods])

    # Document-level prevalence (presence/absence)
    n_before = int(np.sum(y == 0))
    n_after = int(np.sum(y == 1))

    X_bin = (X > 0).astype(int)
    docs_with_before = np.asarray(X_bin[y == 0].sum(axis=0)).ravel()
    docs_with_after = np.asarray(X_bin[y == 1].sum(axis=0)).ravel()

    # proportions with Laplace smoothing
    prop_before = (docs_with_before + 0.5) / (n_before + 1.0)
    prop_after = (docs_with_after + 0.5) / (n_after + 1.0)

    odds_before = prop_before / (1.0 - prop_before)
    odds_after = prop_after / (1.0 - prop_after)
    log_odds = np.log(odds_after) - np.log(odds_before)

    # Chi2 on binary presence
    chi2_stats, pvals = chi2(X_bin, y)
    qvals = fdr_bh(pvals)

    # keep token counts too (for wordcloud sizing if desired)
    counts_before = np.asarray(X[y == 0].sum(axis=0)).ravel()
    counts_after = np.asarray(X[y == 1].sum(axis=0)).ravel()

    # TF-IDF difference (document-level mean)
    tfidf_transformer = TfidfTransformer(norm=None, use_idf=True)
    X_tfidf = tfidf_transformer.fit_transform(X)
    mean_tfidf_before = np.asarray(X_tfidf[y == 0].mean(axis=0)).ravel()
    mean_tfidf_after = np.asarray(X_tfidf[y == 1].mean(axis=0)).ravel()
    tfidf_diff = mean_tfidf_after - mean_tfidf_before

    df_terms = pd.DataFrame({
        "term": terms,
        "docs_with_before": docs_with_before,
        "docs_with_after": docs_with_after,
        "prop_before": prop_before,
        "prop_after": prop_after,
        "count_before": counts_before,
        "count_after": counts_after,
        "log_odds": log_odds,
        "chi2": chi2_stats,
        "pvalue": pvals,
        "qvalue": qvals,
        "tfidf_diff": tfidf_diff
    })

    df_terms["support_docs"] = df_terms["docs_with_before"] + df_terms["docs_with_after"]
    df_terms = df_terms.sort_values(by=["log_odds"], ascending=False).reset_index(drop=True)

    out_csv = os.path.join(out_dir, "word_change_summary.csv")
    df_terms.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n💾 Word change CSV saved: {out_csv}")

    # Print top increased / decreased words (show proportions)
    print("\nTop words enriched AFTER (by log-odds):")
    for _, r in df_terms[df_terms["support_docs"] >= 3].head(n_top).iterrows():
        print(f"{r['term']:<25} after_docs={int(r['docs_with_after']):3}/{n_after} "
              f"before_docs={int(r['docs_with_before']):3}/{n_before} log_odds={r['log_odds']:.2f} q={r['qvalue']:.3g}")

    print("\nTop words enriched BEFORE (by log-odds):")
    for _, r in df_terms[df_terms["support_docs"] >= 3].sort_values("log_odds").head(n_top).iterrows():
        print(f"{r['term']:<25} after_docs={int(r['docs_with_after']):3}/{n_after} "
              f"before_docs={int(r['docs_with_before']):3}/{n_before} log_odds={r['log_odds']:.2f} q={r['qvalue']:.3g}")

    # Determine how many to plot
    plot_n = int(plot_n or n_top)
    try:
        top_pos = df_terms[df_terms["support_docs"] >= 3].head(plot_n)
        top_neg = df_terms[df_terms["support_docs"] >= 3].sort_values("log_odds").head(plot_n)

        height = max(6, plot_n * 0.25)
        fig, ax = plt.subplots(1, 2, figsize=(14, height))
        ax[0].barh(top_pos["term"][::-1], top_pos["log_odds"][::-1], color="tab:green")
        ax[0].set_title("Top words ↑ AFTER (log-odds)")
        ax[1].barh(top_neg["term"][::-1], top_neg["log_odds"][::-1], color="tab:red")
        ax[1].set_title("Top words ↑ BEFORE (log-odds)")

        labelsize = 10 if plot_n <= 30 else (8 if plot_n <= 60 else 6)
        ax[0].tick_params(axis="y", labelsize=labelsize)
        ax[1].tick_params(axis="y", labelsize=labelsize)

        plt.tight_layout()
        plot_path = os.path.join(out_dir, "word_change_barplots.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"💾 Barplots saved: {plot_path}")
    except Exception as e:
        print("Warning: could not create barplots:", e)

    # Wordclouds using document counts (prevalence) for sizing
    try:
        from wordcloud import WordCloud
        after_only = df_terms.sort_values("log_odds", ascending=False).head(100)
        freqs_after = dict(zip(after_only["term"], (after_only["docs_with_after"] + 1)))
        wc_after = WordCloud(width=1200, height=600, background_color="white",
                             collocations=False, prefer_horizontal=0.9)
        wc_after.generate_from_frequencies(freqs_after)
        wc_after_path = os.path.join(out_dir, "wordcloud_after.png")
        wc_after.to_file(wc_after_path)
        print(f"💾 Wordcloud (AFTER) saved: {wc_after_path}")

        before_top = df_terms.sort_values("log_odds", ascending=True).head(100)
        freqs_before = dict(zip(before_top["term"], (before_top["docs_with_before"] + 1)))
        wc_before = WordCloud(width=1200, height=600, background_color="white",
                              collocations=False, prefer_horizontal=0.9)
        wc_before.generate_from_frequencies(freqs_before)
        wc_before_path = os.path.join(out_dir, "wordcloud_before.png")
        wc_before.to_file(wc_before_path)
        print(f"💾 Wordcloud (BEFORE) saved: {wc_before_path}")
    except Exception:
        pass

    return df_terms


if __name__ == "__main__":
    print("\nRunning word-level analysis (före/efter) — saves CSV and plots.")
    # Example: show 60 words in the barplots to make it easier to inspect many terms
    df_word_changes = word_change_analysis(
        texts, periods, stopwords=SWEDISH_STOPWORDS, ngram=(1, 1), n_top=30, plot_n=60, out_dir=BASE_DIR
    )