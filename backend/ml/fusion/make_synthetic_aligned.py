"""Generate a SYNTHETIC aligned multimodal set (text + image + URL per case) -> data/aligned/SYNTHETIC_aligned.csv.

Genuine cases are internally consistent (company, role, salary, domain); scam cases impersonate a known employer.
Noise: missing modalities, subtle scams and slightly odd genuine postings. NOT real evidence.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from ml.common import config
from ml.image import synth_letters
from ml.url.impersonation import load_companies

N_CASES = 240
SCAM_SHARE = 0.4
SALARIES = ["450,000", "600,000", "350,000", "900,000"]
ROLES = ["Data Analyst", "Software Engineer", "Business Analyst", "HR Executive", "Content Writer", "Marketing Intern"]
P_MISSING = {"text": 0.08, "image": 0.35, "url": 0.25}
SUBTLE_SCAM_SHARE = 0.25
ODD_GENUINE_SHARE = 0.10


def _genuine(rng: random.Random, co: str, dom: str, role: str, sal: str) -> tuple[str, str]:
    url = rng.choice([f"https://www.{dom}/careers/apply", f"https://careers.{dom}", f"https://jobs.lever.co/{dom.split('.')[0]}"])
    text = (f"{co} is hiring a {role}. Annual compensation Rs {sal} per annum. Two interview rounds followed by an HR "
            f"discussion. Apply on our careers page or write to careers@{dom}.")
    if rng.random() < ODD_GENUINE_SHARE:
        text += rng.choice([" Immediate joining preferred.", " Apply immediately, limited seats for the campus drive.",
                            " Our recruiter may contact you on WhatsApp for scheduling."])
    return text, url


def _scam(rng: random.Random, co: str, dom: str, role: str, sal: str) -> tuple[str, str]:
    brand = dom.split(".")[0]
    url = rng.choice([f"http://{brand}-hiring.{rng.choice(['online', 'xyz', 'top'])}/apply?id={rng.randint(1, 999)}",
                      f"http://{brand}{rng.randint(10, 99)}.{rng.choice(['site', 'work'])}/register",
                      f"https://{brand}-careers.info/offer", f"http://{brand[:-1]}{brand[-1] * 2}.com/jobs"])
    mail = f"{brand}hr{rng.randint(1, 99)}@gmail.com"
    if rng.random() < SUBTLE_SCAM_SHARE:
        return f"{co} is hiring a {role}. Salary Rs {sal} per annum. Immediate joining. Send your resume to {mail}.", url
    fee = rng.choice(["1,500", "2,999", "999"])
    return (f"Congratulations! You are selected as {role} at {co}. No interview required. Pay refundable registration fee "
            f"Rs {fee} on UPI to confirm your seat. Salary Rs {sal}. Contact HR on WhatsApp only or {mail}."), url


def generate(n: int = N_CASES, seed: int = config.SEED + 3) -> pd.DataFrame:
    """Build the aligned frame (and render case images under data/aligned/images/)."""
    rng, np_rng = random.Random(seed), np.random.default_rng(seed)
    comps = [(c, d[0], b[0]) for c, d, b in load_companies() if c in {"TCS", "Infosys", "Wipro", "Zoho", "Cognizant", "Accenture", "Capgemini"}]
    folder = config.ALIGNED_DIR / "images"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, n + 1):
        label = int(rng.random() < SCAM_SHARE)
        co, dom, _ = rng.choice(comps)
        role, sal = rng.choice(ROLES), rng.choice(SALARIES)
        text, url = (_scam if label else _genuine)(rng, co, dom, role, sal)
        path = folder / f"{config.SYNTHETIC_PREFIX}case_{i:03d}.png"
        synth_letters.make_letter(bool(label), rng, np_rng, company=co, role=role, salary=sal)[0].save(path)
        present = {m: rng.random() > P_MISSING[m] for m in P_MISSING}
        if not any(present.values()):
            present["text"] = True
        rows.append({"case_id": i, "text": text if present["text"] else "",
                     "image_path": str(path.relative_to(config.ROOT_DIR)) if present["image"] else "",
                     "url": url if present["url"] else "", "label": label, "notes": "SYNTHETIC"})
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    """Write data/aligned/SYNTHETIC_aligned.csv."""
    df = generate()
    df.to_csv(config.ALIGNED_DIR / f"{config.SYNTHETIC_PREFIX}aligned.csv", index=False)
    print(f"SYNTHETIC aligned cases: {len(df)} (scam {int(df.label.sum())})")
    return df


if __name__ == "__main__":
    run()
