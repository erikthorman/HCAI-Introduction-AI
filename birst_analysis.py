import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords

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
    "sydnärkes","säby","rönninge","norsjö","degerfors","säby","torg"
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

def read_markdown_files(directory):
    """Läser in alla markdown-filer från en katalog."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

def normalize_lemma(word):
    """Grundform-normalisering för svenska ord."""
    word = word.strip().lower()
    
    # Vanliga ändelser för substantiv
    if len(word) > 6 and word.endswith('ningen'):
        return word[:-6]
    if len(word) > 5 and word.endswith('ning'):
        return word[:-4]
    if len(word) > 4 and word.endswith('en'):
        return word[:-2]
    if len(word) > 3 and word.endswith('n'):
        return word[:-1]
        
    return word

def preprocess_text(text):
    """Förbehandlar text: tar bort skiljetecken, konverterar till lowercase och tar bort stopwords."""
    import re
    import unicodedata

    # Normalisera text (unicode och gemener)
    text = unicodedata.normalize("NFKC", text.lower())

    # Ta bort markdown och länkar
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)

    # Ta bort skiljetecken och siffror
    text = re.sub(r"[^a-zåäö\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Dela upp i ord och normalisera
    words = text.split()
    normalized = [normalize_lemma(w) for w in words]

    # Filtrera bort stoppord
    tokens = [w for w in normalized if w not in SWEDISH_STOPWORDS]

    return tokens


def calculate_birst(documents):
    """Beräknar BIRST term frequency för varje ord."""
    print("Börjar beräkna BIRST...")
    
    # Räkna total frekvens för alla ord i alla dokument
    all_words = []
    for doc in documents:
        words = preprocess_text(doc)
        all_words.extend(words)
        if "omställning" in words or "omställningen" in words:
            print(f"Hittade 'omställning' relaterade ord i dokument. Normaliserade ord:", 
                  [w for w in words if "omställ" in w])
    
    total_freq = Counter(all_words)
    print("\nVanligaste orden:", total_freq.most_common(10))
    
    # Beräkna BIRST för varje dokument
    document_birst = []
    for i, doc in enumerate(documents):
        print(f"\rBearbetar dokument {i+1}/{len(documents)}", end="")
        words = preprocess_text(doc)
        doc_freq = Counter(words)
        
        # Beräkna BIRST för varje ord i dokumentet
        doc_birst = {}
        doc_len = len(words)
        for word, freq in doc_freq.items():
            # BIRST = (ordets frekvens i dokumentet / dokumentets längd) * total frekvens för ordet
            birst = (freq / doc_len) * total_freq[word]
            doc_birst[word] = birst
        
        document_birst.append(doc_birst)
    
    print("\nBIRST-beräkning klar!")
    return document_birst

def main():
    print("Startar analys...")
    
    # Läs in dokument från både före och efter mapparna med absoluta sökvägar
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fore_path = os.path.join(base_dir, "data", "Före Markdown")
    efter_path = os.path.join(base_dir, "data", "Efter Markdown")
    
    print(f"Läser dokument från:\n- {fore_path}\n- {efter_path}")
    fore_docs = read_markdown_files(fore_path)
    efter_docs = read_markdown_files(efter_path)
    print(f"Läste {len(fore_docs)} före-dokument och {len(efter_docs)} efter-dokument")
    
    # Beräkna BIRST för båda dokumentsamlingarna
    print("\nBeräknar BIRST för före-dokument...")
    fore_birst = calculate_birst(fore_docs)
    print("\nBeräknar BIRST för efter-dokument...")
    efter_birst = calculate_birst(efter_docs)
    
    # Kombinera BIRST-värden för alla dokument i varje samling
    print("\nKombinerar BIRST-värden...")
    combined_fore_birst = {}
    for doc_birst in fore_birst:
        for word, value in doc_birst.items():
            combined_fore_birst[word] = combined_fore_birst.get(word, 0) + value

    combined_efter_birst = {}
    for doc_birst in efter_birst:
        for word, value in doc_birst.items():
            combined_efter_birst[word] = combined_efter_birst.get(word, 0) + value
    
    # Skapa DataFrames
    print("\nSkapar DataFrames och plottar...")
    df_fore = pd.DataFrame(list(combined_fore_birst.items()), columns=['Word', 'BIRST_Score'])
    df_fore['Category'] = 'Före'
    df_efter = pd.DataFrame(list(combined_efter_birst.items()), columns=['Word', 'BIRST_Score'])
    df_efter['Category'] = 'Efter'
    
    # Spara till CSV för detaljerad analys
    df_all = pd.concat([df_fore, df_efter])
    csv_path = os.path.join(base_dir, "birst_results.csv")
    df_all.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nDetaljer sparade till: {csv_path}")

    # --- 🔹 1. Skillnader mellan före och efter ---
    df_diff = (
        df_efter.set_index("Word")["BIRST_Score"]
        - df_fore.set_index("Word")["BIRST_Score"]
    )
    df_diff = df_diff.sort_values(ascending=False)
    
    # Skriv ut topp 20 ökningar
    print("\nTopp 20 ord som ökade mest:")
    for word, score in df_diff.head(20).items():
        print(f"{word:<20} {score:>8.2f}")
    
    df_diff = df_diff.head(20).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_diff, x='Word', y='BIRST_Score', color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Ord som ökade mest i BIRST-score efter förändringen')
    plt.tight_layout()
    plot_path = os.path.join(base_dir, "birst_diff_top20.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nPlott sparad till: {plot_path}")

    # --- 🔹 2. Separata topp 10 för före och efter ---
    top_fore = df_fore.nlargest(10, 'BIRST_Score')
    top_efter = df_efter.nlargest(10, 'BIRST_Score')
    df_top = pd.concat([top_fore, top_efter])

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_top, x='Word', y='BIRST_Score', hue='Category')
    plt.xticks(rotation=45, ha='right')
    plt.title('Topp 10 BIRST-ord före och efter')
    plt.tight_layout()
    plot_path2 = os.path.join(base_dir, "birst_top10_fore_efter.png")
    plt.savefig(plot_path2)
    plt.close()
    print(f"Plott sparad till: {plot_path2}")
    
    print("\nAnalys slutförd!")

if __name__ == "__main__":
    main()