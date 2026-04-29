#!/usr/bin/env python3
"""Send PDF page(s) as images to a vLLM /v1/chat/completions endpoint."""

import argparse, base64, sys
from io import BytesIO
from pathlib import Path
import requests
from tqdm.auto import tqdm
import json
import pandas as pd
from glob import glob
import os
import fitz
from PIL import Image

SYSTEM_PROMPT = """\
You are an expert financial data extractor. Extract ALL board director information from the provided annual report page.

LANGUAGE & FORMATTING RULES:
- ALL output text must be in English.
- Translate titles and qualifications.
- Names must be ASCII-only Latin characters.

JSON SCHEMA:
For each director return:
  name (string): Romanised full name
  title (string): Board role (Chair, Director, CEO, etc.)
  gender (string): "male", "female", or null
  appointment_date (string): YYYY-MM-DD or year, or null
  qualifications (list): e.g. ["CPA", "PhD"]
  age_or_birth_year (int): or null
  other_directorships (list): Other company names
  committees (list): [{"name": "Audit", "is_chair": boolean}]
  is_independent (boolean): true/false/null
  is_executive (boolean): true/false/null

Return ONLY a JSON array. If no directors appear, return [].
"""
API_URL = "https://ai.cer-sandbox.cloud.edu.au/v1/chat/completions"
MODEL = "nemotron_3_nano_omni"
MAX_TOKENS = 32000
DPI = 150


def page_to_b64(pdf_path: str, idx: int) -> str:
    doc = fitz.open(pdf_path)
    z = DPI / 72.0
    pix = doc.load_page(idx).get_pixmap(matrix=fitz.Matrix(z, z))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def chat(b64, text):
    r = requests.post(
        API_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": MAX_TOKENS,
            "stream": False,
            "temperature": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    r.raise_for_status()
    return json.loads(
        r.json()["choices"][0]["message"]["content"]
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )


def main():
    pdfs = sorted(glob("data/*.pdf"))

    for pdf in tqdm(pdfs, desc="PDFs"):
        print(f"Processing {pdf}...")
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            print(f"Error opening {pdf}: {e}")
            continue
        n = len(doc)
        doc.close()
        pages = range(n)

        for i in tqdm(pages, desc="Pages", leave=False):
            b64 = page_to_b64(pdf, i)
            for retry in range(3):
                try:
                    result = chat(b64, f"Page {i+1}.\n\n{SYSTEM_PROMPT}")
                    break
                except Exception as e:
                    print(f"Error processing {pdf} page {i+1} (attempt {retry+1}/3): {e}")
                    result = None
            if result:
                print(pd.json_normalize(result))
                os.makedirs("results", exist_ok=True)
                with open(f"results/{Path(pdf).stem}.json", "w") as f:
                    json.dump(result, f, indent=2)
                break


if __name__ == "__main__":
    main()
