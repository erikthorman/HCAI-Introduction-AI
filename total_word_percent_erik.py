#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
from collections import Counter

import pandas as pd


# Minska varning från HuggingFace tokenizers vid multiprocess (behålls om flera skript körs)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")



# Stopwords (svenska) inkl. kommuner/landskap + genitiv-s
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

# --- Svenska kommuner (statisk lista - kept as provided earlier) ---
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
    COMMUNE_STOPWORDS.add(name + "s")

PROVINCE_STOPWORDS = set()
for name in SWEDISH_PROVINCES:
    PROVINCE_STOPWORDS.add(name)
    PROVINCE_STOPWORDS.add(name + "s")

SWEDISH_STOPWORDS.update(COMMUNE_STOPWORDS)
SWEDISH_STOPWORDS.update(PROVINCE_STOPWORDS)






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
        percent_point_change = (freq_after - freq_before) * 100
        relative_change = ((freq_after - freq_before) / freq_before * 100) if freq_before > 0 else None
        rows.append({
            "word": word,
            "freq_before_percent": freq_before * 100,
            "freq_after_percent": freq_after * 100,
            "percent_point_change": percent_point_change,
            "relative_percent_change": relative_change
        })

    df_change = pd.DataFrame(rows)
    df_change = df_change.sort_values("percent_point_change", ascending=False).head(top_n).reset_index(drop=True)
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
