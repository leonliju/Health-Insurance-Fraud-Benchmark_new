#!/usr/bin/env python3
"""
Optional: rebuild reference/hbp_2_2_master_extracted.csv from the official NHA master PDF
(National Master Health Benefit Packages 2.2). Needs: pip install pdfplumber

Usage: python src/extract_hbp.py --pdf National_Master_Health_Benefit_packages_2_2.pdf --out reference/hbp_2_2_master_extracted.csv
The extraction is automatic; spot-check against the PDF before quoting a number.
"""
import argparse

import pandas as pd
import pdfplumber

COLS = ["page", "spec_code", "specialty", "package_code", "package_name", "procedure_code", "procedure_name",
        "procedure_price", "stratification", "implant", "total_price", "label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default="reference/hbp_2_2_master_extracted.csv")
    a = ap.parse_args()
    rows = []
    with pdfplumber.open(a.pdf) as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            for table in page.extract_tables():
                for r in table:
                    if r and r[0] and r[0].startswith("Specialty Code"):
                        continue
                    rows.append([pno] + [("" if c is None else str(c).replace("\n", " ").strip()) for c in r])
    width = len(COLS)
    rows = [(r + [""] * width)[:width] for r in rows]
    pd.DataFrame(rows, columns=COLS).to_csv(a.out, index=False)
    print(f"{len(rows)} procedure rows written to {a.out}")


if __name__ == "__main__":
    main()
