import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

# --- Stopwords från analysis_stine.py ---
nltk.download("stopwords", quiet=True)
SWEDISH_STOPWORDS = set(stopwords.words("swedish"))

EXTRA_STOPWORDS = {
    "varav","toverud","utveckl","bengtsfa","gunn","guid","stahl","pris","ocksa","hööra","tjör",
    "unikomfamilj","svalöva","degerfa","årjängas","norrbott","olofströms","gislaveda","nykvar",
    "emmabod","uppsal","åstorpa","gölisk","svedal","älmhult","itd","munkfor","munkfa","sydnärke",
    "kungsöra","sandvik","årjänga","österåkers","ska","stockholmarna","kristianstads","karlstads",
    "kommer","kommun","kommuner","kommunens","emmaboda","vännäs","sätt","rätt","genom","kommunkoncernen",
    "samt","image","kr","nok","pa","mom","ekerö","älmhults","lsa","göliska","eblomlådan","stockholmarnas",
    "sydnärkes","säby","rönninge","norsjö","degerfors","säby","torg", "www"
}
SWEDISH_STOPWORDS.update(EXTRA_STOPWORDS)

# --- Svenska kommuner ---
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

# --- Lägg till kommuner och landskap till stopwords ---
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


# --- Function definitions ---
def read_markdown_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

def preprocess_text(text):
    import re
    import unicodedata
    text = unicodedata.normalize("NFKC", text.lower())
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-zåäö\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    tokens = [w for w in words if w not in SWEDISH_STOPWORDS]
    return tokens

def calculate_birst(documents):
    print("Calculating BIRST scores...")
    all_words = []
    for doc in documents:
        words = preprocess_text(doc)
        all_words.extend(words)
    total_freq = Counter(all_words)
    document_birst = []
    for i, doc in enumerate(documents):
        words = preprocess_text(doc)
        doc_freq = Counter(words)
        doc_birst = {}
        doc_len = len(words)
        for word, freq in doc_freq.items():
            birst = (freq / doc_len) * total_freq[word]
            doc_birst[word] = birst
        document_birst.append(doc_birst)
    print("BIRST calculation completed!")
    return document_birst

def merge_words_in_df(df, merge_dict):
    df = df.copy()
    df['Word'] = df['Word'].apply(lambda w: merge_dict.get(w, w))
    df = df.groupby('Word', as_index=False)['BIRST_Score'].sum()
    return df

def save_birst_visualizations(df_fore, df_after, base_dir, top_n=30):
    merged = pd.merge(df_fore, df_after, on="Word", suffixes=("_Before", "_After"))
    merged['BIRST_Diff'] = merged['BIRST_Score_After'] - merged['BIRST_Score_Before']
    merged['Total_BIRST'] = merged['BIRST_Score_After'] + merged['BIRST_Score_Before']

    # --- HEATMAP: Words with biggest change ---
    top_words = merged.nlargest(top_n, 'BIRST_Diff')['Word']

    # Tvinga heatmap_df att följa samma ordning som top_words
    heatmap_df = merged.loc[merged['Word'].isin(top_words)]
    heatmap_df = heatmap_df.set_index('Word')[['BIRST_Score_Before', 'BIRST_Score_After']]
    heatmap_df = heatmap_df.loc[top_words]  # <-- behåller top_words-ordningen

    plt.figure(figsize=(12,10))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f"Top {top_n} Words with Largest Increase in BIRST Score", fontsize=14, pad=10)
    plt.xlabel("Time Period")
    plt.ylabel("Word")
    heatmap_path = os.path.join(base_dir, 'birst_top_words_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()


    # --- BAR CHART: Words with highest BIRST After 2023 ---
    top_after = df_after.nlargest(top_n, 'BIRST_Score')
    plt.figure(figsize=(12,8))
    sns.barplot(data=top_after, x='BIRST_Score', y='Word', palette='viridis')
    plt.title(f"Top {top_n} Words with Highest BIRST Score (After 2023)", fontsize=14, pad=10)
    plt.xlabel("BIRST Score (After 2023)")
    plt.ylabel("Word")
    bar_path = os.path.join(base_dir, 'birst_top_after_barchart.png')
    plt.savefig(bar_path, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved as PNG:\n- {heatmap_path}\n- {bar_path}")

# --- Main ---
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fore_path = os.path.join(base_dir, "data", "Före Markdown")
    after_path = os.path.join(base_dir, "data", "Efter Markdown")

    fore_docs = read_markdown_files(fore_path)
    after_docs = read_markdown_files(after_path)

    fore_birst = calculate_birst(fore_docs)
    after_birst = calculate_birst(after_docs)

    combined_fore_birst = {}
    for doc_birst in fore_birst:
        for word, value in doc_birst.items():
            combined_fore_birst[word] = combined_fore_birst.get(word, 0) + value

    combined_after_birst = {}
    for doc_birst in after_birst:
        for word, value in doc_birst.items():
            combined_after_birst[word] = combined_after_birst.get(word, 0) + value

    df_fore = pd.DataFrame(list(combined_fore_birst.items()), columns=['Word', 'BIRST_Score'])
    df_fore['Category'] = 'Before'
    df_after = pd.DataFrame(list(combined_after_birst.items()), columns=['Word', 'BIRST_Score'])
    df_after['Category'] = 'After'

    # --- Manuella sammanslagningar ---
    MERGE_WORDS = {
        "socialförvaltningen": "socialförvaltning",
        "socialförvaltningens": "socialförvaltning",
        "omställningen" : "omställning",
        "programmets": "programmet",
        "programmen": "programmet",
        "digitaliseringsresan": "digitaliseringsresa",
        "digitaliseringsarbetet": "digitaliseringsarbetets",
        "verksamhetsutvecklingen": "verksamhetsutveckling",
        "förvaltningens": "förvaltning",
        "förvaltningar": "förvaltning",
        "planen" : "plan",
        "strategins": "strategi",
        "strategier": "strategi",
        "lösningarna": "lösning",
        "lösningars": "lösning",
        "tjänsterna": "tjänst",
        "tjänsters": "tjänst",
        "möjligheterna": "möjlighet",
        "möjligheters": "möjlighet",
        "utmaningarna": "utmaning",
        "utmaningars": "utmaning",
        "förbättringarna": "förbättring",
        "förbättringars": "förbättring",
        "användarna": "användare",
        "användares": "användare",
        "digitaliseringens": "digitalisering",
        "digitaliseringars": "digitalisering",
        "invånarnas": "invånare",
        "kommunens": "kommun",
        "kommuners": "kommun",
        "medborgarnas": "medborgare",
        "medborgare": "medborgare",
        "möjliggör": "möjliggöra",
        "påverkar": "påverka",
        "förbättrar": "förbättra",
        "utvecklar": "utveckla",
        "använder": "använda",
        "digitaliserar": "digitalisera",
        # fler ord kan läggas till här
    }
    df_fore = merge_words_in_df(df_fore, MERGE_WORDS)
    df_after = merge_words_in_df(df_after, MERGE_WORDS)

    df_all = pd.concat([df_fore, df_after])
    df_all.to_csv(os.path.join(base_dir, "birst_results.csv"), index=False, encoding='utf-8')

    df_diff = df_after.set_index("Word")["BIRST_Score"] - df_fore.set_index("Word")["BIRST_Score"]
    df_diff = df_diff.sort_values(ascending=False)
    print("\nTop 20 words with the largest increase in BIRST:")
    print(df_diff.head(20))

    save_birst_visualizations(df_fore, df_after, base_dir, top_n=30)

if __name__ == "__main__":
    main()
