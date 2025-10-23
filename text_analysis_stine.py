import os
import glob
from collections import Counter
import re
from pathlib import Path
import matplotlib.pyplot as plt

# --- NLTK stoppord (svenska) ---
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)
SWEDISH_STOPWORDS = set(stopwords.words("swedish"))

# +++ NYTT: normalisering och stemming +++
import unicodedata
from nltk.stem.snowball import SnowballStemmer
STEMMER = SnowballStemmer("swedish")

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

# (Valfritt) lägg till egna stoppord om du vill filtrera mer
SWEDISH_STOPWORDS.update({"ska", "kommer", "kan", "kommun"})

# --- Sökvägar – robusta relativt skriptets fil ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEFORE_GLOB = os.path.join(BASE_DIR, "data", "Före Markdown", "*.md")
AFTER_GLOB  = os.path.join(BASE_DIR, "data", "Efter Markdown", "*.md")

# --- Hjälpfunktioner ---
def normalize_text(s: str) -> str:
    """Unicode-normalisera (NFKC) för att undvika att olika teckenvarianter blir olika ord."""
    return unicodedata.normalize("NFKC", s)

def load_texts(path_pattern: str):
    texts = []
    files = sorted(glob.glob(path_pattern))
    for filepath in files:
        text = Path(filepath).read_text(encoding="utf-8")
        texts.append(normalize_text(text))  # normalisera här
    return texts

def tokenize(text: str):
    # behåll a–ö, kasta siffror/tecken, gör gemener
    words = re.findall(r"\b[a-zA-ZåäöÅÄÖ]+\b", text.lower())
    # filtrera bort stoppord
    return [w for w in words if w not in SWEDISH_STOPWORDS]

def count_words(texts):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    return counter

# +++ NYTT: stamräkning och ytforms-mappar +++
def stem_counts(surface_counter: Counter):
    """
    Summera counts per stam och bygg en mapping stam -> Counter(ytsformer)
    så att vi kan visa 'representativ' ytform (den vanligaste).
    """
    stem_counter = Counter()
    stem_to_surface = {}
    for surface, c in surface_counter.items():
        stem = STEMMER.stem(surface)
        stem_counter[stem] += c
        d = stem_to_surface.setdefault(stem, Counter())
        d[surface] += c
    return stem_counter, stem_to_surface

def rep_surface(stem: str, mapping: dict):
    """Hämta mest frekventa ytformen för en given stam."""
    return (mapping[stem].most_common(1)[0][0]) if stem in mapping else stem

def sample_items(items, n=40):
    """Jämnt urplock av items över hela listan (alfabetiskt sorterad)."""
    if len(items) <= n:
        return items
    step = max(1, len(items) // n)
    return [items[i] for i in range(0, len(items), step)][:n]

# --- Läs in data ---
before_texts = load_texts(BEFORE_GLOB)
after_texts  = load_texts(AFTER_GLOB)

if not before_texts:
    print(f"⚠️ Hittade inga filer för 'före': {BEFORE_GLOB}")
if not after_texts:
    print(f"⚠️ Hittade inga filer för 'efter': {AFTER_GLOB}")

# --- Räkna ordfrekvenser (ytformer) ---
before_counts = count_words(before_texts)
after_counts  = count_words(after_texts)

# --- Konvertera till stam-nivå + bygg ytformsmappar ---
before_stem_counts, before_stem_to_surface = stem_counts(before_counts)
after_stem_counts,  after_stem_to_surface  = stem_counts(after_counts)

before_stems = set(before_stem_counts.keys())
after_stems  = set(after_stem_counts.keys())

# --- Nya och borttagna på stam-nivå ---
new_stems     = sorted(after_stems - before_stems)
removed_stems = sorted(before_stems - after_stems)

# --- Frekvensförändring (efter/före) på stam-nivå ---
MIN_AFTER_COUNT = 1  # ändra till t.ex. 3 för att filtrera brus
freq_change_stem = {
    s: after_stem_counts[s] / (before_stem_counts.get(s, 0) + 1)
    for s in after_stems
    if after_stem_counts[s] >= MIN_AFTER_COUNT
}
most_increased_stems = sorted(freq_change_stem.items(), key=lambda x: x[1], reverse=True)[:20]

# --- Utskrift (visa representativa ytformer) ---
print("— Sammanfattning (stam-nivå) —")
print(f"Antal unika stammar FÖRE: {len(before_stems)}")
print(f"Antal unika stammar EFTER: {len(after_stems)}")
print(f"Nya stammar EFTER (inte sedda FÖRE): {len(new_stems)}")
print(f"Borttagna stammar EFTER (fanns FÖRE men inte EFTER): {len(removed_stems)}")
print()

print("Exempel på nya ord (urval, representativ ytform):")
new_examples = [rep_surface(s, after_stem_to_surface) for s in sample_items(new_stems, n=40)]
print(", ".join(new_examples))
print()

print("Exempel på borttagna ord (urval, representativ ytform):")
removed_examples = [rep_surface(s, before_stem_to_surface) for s in sample_items(removed_stems, n=40)]
print(", ".join(removed_examples))
print()

print("— Topp 20 som ökat mest (efter/före), visar representativ ytform —")
for stem, ratio in most_increased_stems:
    print(f"{rep_surface(stem, after_stem_to_surface):<20} {ratio:.2f}×")

# --- (Valfritt) Visualisera toppökningar ---
if most_increased_stems:
    top_words = [rep_surface(s, after_stem_to_surface) for s, _ in most_increased_stems[:20]]
    top_ratios = [r for _, r in most_increased_stems[:20]]
    plt.figure()
    plt.barh(top_words, top_ratios)
    plt.xlabel("Förändringsfaktor (efter/före)")
    plt.title("Begrepp (stammar) som ökat mest efter januari 2023")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# --- Exportera till Excel (flera flikar) ---
import pandas as pd
output_path = os.path.join(BASE_DIR, "word_analysis_results.xlsx")

def row_for_stem(stem: str):
    """Bygger en rad med frekvenser och representativ ytform(er) för en given stam."""
    cb = int(before_stem_counts.get(stem, 0))
    ca = int(after_stem_counts.get(stem, 0))
    # välj rep ytform från respektive period (bra för jämförelse)
    rep_b = rep_surface(stem, before_stem_to_surface)
    rep_a = rep_surface(stem, after_stem_to_surface)
    # statusfält
    if cb == 0 and ca > 0:
        status = "new"
    elif cb > 0 and ca == 0:
        status = "removed"
    else:
        status = "both"
    ratio = (ca / cb) if cb > 0 else (float("inf") if ca > 0 else 0.0)
    return {
        "stem": stem,
        "rep_surface_before": rep_b if stem in before_stem_to_surface else None,
        "rep_surface_after":  rep_a if stem in after_stem_to_surface  else None,
        "count_before": cb,
        "count_after": ca,
        "ratio_after_over_before": ratio,
        "status": status,
    }

# --- Blad 1: nya ord (med frekvenser) ---
df_new = pd.DataFrame(
    [
        {
            "stem": s,
            "rep_surface": rep_surface(s, after_stem_to_surface),
            "count_before": int(before_stem_counts.get(s, 0)),
            "count_after": int(after_stem_counts.get(s, 0)),
            "ratio_after_over_before": float("inf")
            if before_stem_counts.get(s, 0) == 0
            else after_stem_counts[s] / before_stem_counts[s],
        }
        for s in sorted(new_stems)
    ]
)

# --- Blad 2: borttagna ord (med frekvenser) ---
df_removed = pd.DataFrame(
    [
        {
            "stem": s,
            "rep_surface": rep_surface(s, before_stem_to_surface),
            "count_before": int(before_stem_counts.get(s, 0)),
            "count_after": 0,
            "ratio_after_over_before": 0.0,
        }
        for s in sorted(removed_stems)
    ]
)

# --- Blad 3: sammanfattning över ALLA stammar ---
all_stems_sorted = sorted(before_stems | after_stems)
df_summary = pd.DataFrame([row_for_stem(s) for s in all_stems_sorted])



with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df_new.to_excel(writer, sheet_name="new_words", index=False)
    df_removed.to_excel(writer, sheet_name="removed_words", index=False)
    df_summary.to_excel(writer, sheet_name="summary_all_stems", index=False)
    

print(f"\n💾 Resultaten har sparats till: {output_path}")
print("Flikar: 'new_words', 'removed_words', 'summary_all_stems', 'top_increased'")