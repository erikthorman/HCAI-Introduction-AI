#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Börja med att skapa en virtuell miljö och installera dependencies:

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install docling docling-core tqdm pandas






Batch-extrahera text & struktur ur många PDF:er med Docling.
- Reproducerar din "single document"-pipeline automatiskt för en hel mapp.
- Skapar .json och/eller .md per PDF + index.csv.

Exempel:
pip install -U docling docling-core tqdm pandas
python batch_docling_extract.py --in_dir "/sökväg/till/strategier" --out_dir "/sökväg/till/output" --format both


python batch_docling_extract.py \
  --in_dir "/Users/erikthorman/Downloads/digital/Efter 2023.01.01" \
  --out_dir "/Users/erikthorman/Downloads/digital/efter 2023 bearbetad" \
  --format both --recursive


python batch_docling_extract.py \
  --in_dir "/Users/erikthorman/Downloads/digital/Före 2023.01.01" \
  --out_dir "/Users/erikthorman/Downloads/digital/Före 2023 Extraherad Text" \
  --format both --recursive


"""

import os
import re
import sys
import json
import time
import argparse
import warnings
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# Minska brus
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Docling
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
# Typ: from docling_core.types.doc import DoclingDocument  # endast för type hints (ej nödvändigt)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("batch_docling")


# -----------------------------
# Hjälpfunktioner (från din notebook, anpassade)
# -----------------------------

def get_document_statistics(doc_dict: dict) -> dict:
    """Räkna sidor, sektioner, tabeller, mm (som i din notebook)."""
    stats = {
        "num_pages": len(doc_dict.get("pages", {})) if "pages" in doc_dict else 1,
        "num_words": 0,
        "num_tables": 0,
        "num_figures": 0,
        "num_section_headers": 0,
        "num_page_headers": 0,
        "num_footers": 0,
    }

    for element in doc_dict.get("elements", []):
        element_type = str(element.get("type", "")).lower()

        if element_type == "page_header":
            stats["num_page_headers"] += 1
        elif element_type == "section_header":
            stats["num_section_headers"] += 1
        elif element_type == "table":
            stats["num_tables"] += 1
        elif element_type in ["figure", "image"]:
            stats["num_figures"] += 1
        elif element_type == "footer":
            stats["num_footers"] += 1

        if "text" in element and element["text"]:
            stats["num_words"] += len(element["text"].split())

    return stats


def convert_document_to_dict(doc) -> dict:
    """Konvertera DoclingDocument till ett enkelt dict (som i din notebook)."""
    document_dict = {
        "name": getattr(doc, "name", None),
        "pages": {},
        "elements": []
    }

    # Sidor
    try:
        for page_num, page in doc.pages.items():
            document_dict["pages"][str(page_num)] = {
                "number": page_num,
                "size": {
                    "width": getattr(getattr(page, "size", None), "width", None),
                    "height": getattr(getattr(page, "size", None), "height", None),
                },
            }
    except Exception:
        pass

    # Element
    try:
        for item, level in doc.iterate_items():
            element_dict = {
                "type": getattr(item, "label", None),
                "level": level,
                "page_numbers": [getattr(p, "page_no", None) for p in getattr(item, "prov", [])],
                "bounding_boxes": [
                    getattr(getattr(p, "bbox", None), "as_tuple", lambda: None)()
                    for p in getattr(item, "prov", [])
                    if getattr(p, "bbox", None) is not None
                ],
            }
            if hasattr(item, "text"):
                element_dict["text"] = item.text
            document_dict["elements"].append(element_dict)
    except AttributeError:
        # iterate_items saknas i vissa edge-fall
        pass

    return document_dict


def slugify(name: str) -> str:
    """Säkra filnamn."""
    name = re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:180]


def process_single_file(pdf_path: Path, output_dir: Path, format_choice: str = "both") -> dict:
    """
    Kör din Docling-pipeline på EN fil:
      - OCR avstängt (som i din kod)
      - returnerar dokument-dict (för ev. debug / visning)
      - skriver .json/.md till output_dir beroende på 'format_choice'
    """
    start_time = datetime.now()

    # Pipeline-options: samma som i din notebook
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.do_ocr = False  # DISABLE OCR (som din kod)

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Konvertera
    conversion_result = doc_converter.convert(str(pdf_path))
    doc = conversion_result.document
    if not doc or not getattr(doc, "pages", None):
        raise ValueError(f"No valid pages in {pdf_path.name}")

    # Till dict + statistik
    doc_dict = convert_document_to_dict(doc)
    stats = get_document_statistics(doc_dict)

    # Metadata
    doc_dict["metadata"] = {
        "filename": pdf_path.name,
        "path": str(pdf_path),
        "processing_time": str(datetime.now() - start_time),
        "page_count": len(doc.pages),
        "extraction_timestamp": datetime.now().isoformat(),
        "statistics": stats,
    }

    # Spara ut
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(pdf_path.stem)

    out_json = output_dir / f"{stem}_docling.json"
    out_md   = output_dir / f"{stem}_docling.md"

    if format_choice in ("json", "both"):
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(doc_dict, f, ensure_ascii=False, indent=2)

    if format_choice in ("markdown", "both"):
        markdown_content = doc.export_to_markdown()
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    return {
        "doc_dict": doc_dict,
        "stats": stats,
        "out_json": str(out_json) if out_json.exists() else None,
        "out_md": str(out_md) if out_md.exists() else None,
    }


# -----------------------------
# CLI / Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch-extrahera PDF:er med Docling (JSON/Markdown).")
    parser.add_argument("--in_dir", type=str, required=True, help="Mapp med PDF-filer.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output-mapp för JSON/MD/index.csv.")
    parser.add_argument("--format", type=str, default="both", choices=["json", "markdown", "both"],
                        help="Vilka filer som ska genereras per PDF.")
    parser.add_argument("--recursive", action="store_true", help="Sök rekursivt efter PDF:er.")
    parser.add_argument("--pattern", type=str, default="*.pdf", help="Filglob (default: *.pdf).")
    args = parser.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"❌ IN_DIR saknas: {in_dir}")
        sys.exit(1)

    # Hitta pdf:er
    if args.recursive:
        pdf_paths = sorted(in_dir.rglob(args.pattern))
    else:
        pdf_paths = sorted(in_dir.glob(args.pattern))

    pdf_paths = [p for p in pdf_paths if p.suffix.lower() == ".pdf"]

    if not pdf_paths:
        print(f"⚠️ Hittade inga PDF:er i {in_dir} (pattern: {args.pattern}, recursive={args.recursive}).")
        sys.exit(0)

    print(f"▶️  Bearbetar {len(pdf_paths)} PDF:er från: {in_dir}")
    rows = []

    for pdf_path in tqdm(pdf_paths, desc="Processing"):
        t0 = time.time()
        try:
            res = process_single_file(pdf_path, out_dir, args.format)
            stats = res.get("stats", {}) or {}
            rows.append({
                "filename": pdf_path.name,
                "input_path": str(pdf_path),
                "output_json": res.get("out_json"),
                "output_md": res.get("out_md"),
                "pages": stats.get("num_pages"),
                "words": stats.get("num_words"),
                "tables": stats.get("num_tables"),
                "figures": stats.get("num_figures"),
                "section_headers": stats.get("num_section_headers"),
                "elapsed_sec": round(time.time() - t0, 2),
                "status": "ok",
            })
        except Exception as e:
            logger.exception(f"Fel vid bearbetning av {pdf_path.name}")
            rows.append({
                "filename": pdf_path.name,
                "input_path": str(pdf_path),
                "output_json": None,
                "output_md": None,
                "pages": None,
                "words": None,
                "tables": None,
                "figures": None,
                "section_headers": None,
                "elapsed_sec": round(time.time() - t0, 2),
                "status": f"error: {e}",
            })

    # Skriv index.csv
    df = pd.DataFrame(rows)
    index_csv = out_dir / "index.csv"
    df.to_csv(index_csv, index=False, encoding="utf-8")

    n_ok = (df["status"] == "ok").sum()
    print("\n✅ Klart!")
    print(f"   OK: {n_ok} / {len(df)}")
    print(f"   Index: {index_csv}")
    print(f"   Output: {out_dir}")

if __name__ == "__main__":
    main()
