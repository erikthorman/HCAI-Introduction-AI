import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Stoppord (svenska) inkl. kommuner/landskap + genitiv-s
import nltk, re, unicodedata
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)
SWEDISH_STOPWORDS = set(stopwords.words("swedish"))

# (valfritt) lägg in kommuner/landskap + "s" från din tidigare lista om du vill
# Här visar jag bara hur du kan hooka in egna stoppord:
EXTRA_STOPWORDS = {"varav","toverud","utveckl","bengtsfa","gunn","guid","stahl","pris","ocksa","hööra","tjör","unikomfamilj","svalöva","degerfa","årjängas","norrbott", "olofströms","gislaveda","nykvar","emmabod","uppsal","åstorpa","gölisk","svedal","älmhult","itd","munkfor","munkfa","sydnärke","kungsöra","sandvik","årjänga","österåkers","ska", "stockholmarna", "kristianstads","karlstads","kommer", "kommun", "kommuner", "kommunens", "emmaboda", "vännäs", "sätt", "rätt", "genom", "kommunkoncernen", "samt", "image" ,"kr", "nok","pa","mom","ekerö","älmhults","lsa","göliska","eblomlådan","stockholmarnas","sydnärkes","säby", "rönninge","norsjö","degerfors","säby","torg"
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

def preprocess_text(text: str, lemmatize: bool = True) -> str:
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

def load_markdown_texts(path_pattern, period_label, lemmatize=True):
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


texts_before, files_before, labels_before = load_markdown_texts(BEFORE_GLOB, "före")
texts_after,  files_after,  labels_after  = load_markdown_texts(AFTER_GLOB, "efter")

texts = texts_before + texts_after
files = files_before + files_after
periods = labels_before + labels_after

print(f"Totalt {len(texts)} dokument lästa ({len(texts_before)} före, {len(texts_after)} efter).")

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

def word_change_analysis(texts, periods, stopwords=SWEDISH_STOPWORDS, ngram=(1,1), n_top=25, out_dir=BASE_DIR):
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


    # Create a wordcloud for AFTER and BEFORE top terms
    try:
        from wordcloud import WordCloud
        after_only = df_terms.sort_values("log_odds", ascending=False).head(100)
        freqs = dict(zip(after_only["term"], (after_only["count_after"]+1)))
        wc = WordCloud(width=1200, height=600, background_color="white", collocations=False, prefer_horizontal=0.9)
        wc.generate_from_frequencies(freqs)
        wc_path = os.path.join(out_dir, "wordcloud_after.png")
        wc.to_file(wc_path)
        print(f"💾 Wordcloud (AFTER) saved: {wc_path}")

        # Top BEFORE-enriched terms
        before_top = df_terms.sort_values("log_odds", ascending=True).head(100)
        freqs_before = dict(zip(before_top["term"], (before_top["count_before"] + 1)))
        wc_before = WordCloud(width=1200, height=600, background_color="white", collocations=False, prefer_horizontal=0.9)
        wc_before.generate_from_frequencies(freqs_before)
        wc_before_path = os.path.join(out_dir, "wordcloud_before.png")
        wc_before.to_file(wc_before_path)
        print(f"💾 Wordcloud (BEFORE) saved: {wc_before_path}")
    except Exception:
        # silently skip if library missing or other errors
        pass

    return df_terms

# Run the analysis using your existing texts & periods and stopwords
print("\nKör ord-analys (före/efter) — detta sparar CSV och plottar top words.")
df_word_changes = word_change_analysis(texts, periods, stopwords=SWEDISH_STOPWORDS, ngram=(1,1), n_top=30)