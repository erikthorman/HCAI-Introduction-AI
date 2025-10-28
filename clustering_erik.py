#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
from collections import Counter

import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

# Minska varning från HuggingFace tokenizers vid multiprocess (behålls om flera skript körs)
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

    # manuella stopwords
    "varav", "toverud", "utveckl", "bengtsfa", "gunn", "guid", "stahl", "pris", "ocksa", "hööra", "tjör", "unikomfamilj",
    "svalöva", "degerfa", "årjängas", "norrbott", "olofströms", "gislaveda", "nykvar", "emmabod", "uppsal", "åstorpa",
    "gölisk", "svedal", "älmhult", "itd", "munkfor", "munkfa", "sydnärke", "kungsöra", "sandvik", "årjänga", "österåkers",
    "ska", "stockholmarna", "kristianstads", "karlstads", "kommer", "kommun", "kommuner", "kommunens", "emmaboda",
    "vännäs", "sätt", "rätt", "genom", "kommunkoncernen", "samt", "image", "kr", "nok", "pa", "mom", "ekerö", "älmhults",
    "lsa", "göliska", "eblomlådan", "stockholmarnas", "sydnärkes", "säby", "rönninge", "norsjö", "degerfors", "säby", "torg"
}

# NLTK:s svenska stoppord
NLTK_SWEDISH = set(stopwords.words("swedish"))

# Sammanfoga NLTK + manuella
SWEDISH_STOPWORDS = set(NLTK_SWEDISH) | set(MANUAL_EXTRA_STOPWORDS)


########################################
# Hjälpfunktioner
########################################

def read_markdown_dir(dir_path, period_label):
    """Läs alla .md-filer rekursivt."""
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
    """Rensa text: små bokstäver, ta bort siffror, specialtecken."""
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


def compute_word_change(df, stop_words, top_n):
    """Beräkna procentuell förändring i ordfrekvens mellan perioder."""
    before_texts = df[df["period"] == "before"]["clean_text"].tolist()
    after_texts = df[df["period"] == "after"]["clean_text"].tolist()

    def count_words(texts):
        counts = Counter()
        for text in texts:
            for word in text.split():
                if word not in stop_words and len(word) > 2:
                    counts[word] += 1
        return counts

    before_counts = count_words(before_texts)
    after_counts = count_words(after_texts)

    total_before = sum(before_counts.values()) or 1
    total_after = sum(after_counts.values()) or 1

    before_freq = {word: before_counts[word] / total_before for word in before_counts}
    after_freq = {word: after_counts[word] / total_after for word in after_counts}

    all_words = set(before_counts) | set(after_counts)
    rows = []
    for word in all_words:
        freq_before = before_freq.get(word, 0.0)
        freq_after = after_freq.get(word, 0.0)
        percent_change = (freq_after - freq_before) * 100
        rows.append({
            "word": word,
            "freq_before": freq_before * 100,
            "freq_after": freq_after * 100,
            "percent_change": percent_change
        })

    df_change = pd.DataFrame(rows)
    df_change = df_change.sort_values("percent_change", ascending=False).head(top_n).reset_index(drop=True)
    return df_change


def parse_args():
    parser = argparse.ArgumentParser(description="Beräkna procentuell ordförändring mellan två perioder.")
    parser.add_argument("--before_dir", required=True, help="Sökväg till katalog med dokument före perioden.")
    parser.add_argument("--after_dir", required=True, help="Sökväg till katalog med dokument efter perioden.")
    parser.add_argument("--out_csv", help="Sökväg till CSV-filen som ska skapas.")
    parser.add_argument("--out_dir", help="Katalog att spara resultatet i (skapar word_shift.csv).")
    parser.add_argument("--top_n", type=int, default=100, help="Antal ord att inkludera i resultatet.")

    args = parser.parse_args()
    if not args.out_csv and not args.out_dir:
        parser.error("ange --out_csv eller --out_dir")
    return args


def main():
    args = parse_args()
    stop_words = get_swe_stopwords()

    print("1) Läser och städar text ...")
    docs_before = read_markdown_dir(args.before_dir, "before")
    docs_after = read_markdown_dir(args.after_dir, "after")
    df = pd.DataFrame(docs_before + docs_after)
    df["clean_text"] = df["text"].apply(clean_text)
    print(f"   Totalt {len(df)} dokument")

    print("2) Räknar procentuell ordförändring ...")
    word_change_df = compute_word_change(df, stop_words, args.top_n)

    if args.out_csv:
        out_path = args.out_csv
    else:
        out_path = os.path.join(args.out_dir, "word_shift.csv")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    word_change_df.to_csv(out_path, index=False, encoding="utf-8")

    print("\n✅ Klart! Resultat:")
    print(f"- {out_path}")


if __name__ == "__main__":
    main()
