"""SYNTHETIC data generators used when real datasets are not yet in data/raw/.

Everything produced here is fake and must be labelled SYNTHETIC (filenames start with
``config.SYNTHETIC_PREFIX``). Results computed on it are NOT scientific evidence.
"""
from __future__ import annotations

import random

import pandas as pd

from ml.common import config

GENUINE_TITLES = [
    "Software Engineer", "Data Analyst", "Marketing Manager", "Customer Support Associate",
    "Accountant", "Mechanical Engineer", "UX Designer", "HR Coordinator", "Sales Executive",
    "DevOps Engineer", "Content Writer", "Business Analyst", "Project Manager", "QA Tester",
]
SCAM_TITLES = [
    "Work From Home Data Entry", "Online Typist Urgent", "Part Time Earn Daily", "Packing Job At Home",
    "Offer Letter Process Executive", "Ad Posting Job", "Mobile App Tester Earn Per Task",
    "Customer Care Executive Immediate Joining", "Software Engineer", "Marketing Executive",
]
COMPANIES = ["Northwind", "Helios Systems", "BlueOrbit", "Kestrel Labs", "Aurora Retail", "Vertex Analytics"]
SKILLS = ["Python", "SQL", "Excel", "communication", "teamwork", "Java", "problem solving", "Tableau", "React"]
G_DESC = [
    "We are looking for a motivated {t} to join our growing team.",
    "You will collaborate with cross-functional teams to deliver high quality results.",
    "Responsibilities include planning, execution and reporting of key deliverables.",
    "The role reports to the head of department and works closely with stakeholders.",
    "Our team values ownership, learning and transparent communication.",
    "Candidates will go through two interview rounds followed by a technical assessment.",
    "This is a full time position based in our {city} office with hybrid flexibility.",
    "Immediate joining is a plus but not mandatory. Apply now through our careers page.",
    "Competitive salary as per industry standards and performance based bonus.",
]
CITIES = ["Bengaluru", "Hyderabad", "Pune", "Chennai", "Mumbai", "Delhi", "Austin", "London", "Toronto"]
G_BEN = ["Health insurance", "Paid time off", "Learning budget", "Flexible hours", "Provident fund", "Free lunch"]
S_DESC_OBVIOUS = [
    "No experience needed! Earn up to {amt} per week working from home just 2 hours a day.",
    "Selection is guaranteed, there is no interview. Joining within 24 hours.",
    "Pay a refundable registration fee of {fee} to confirm your seat and receive the offer letter.",
    "Send your Aadhaar, PAN and bank account details to complete verification.",
    "Contact our HR on WhatsApp only. Limited seats, apply immediately before the offer expires.",
    "Training kit charges {fee} to be paid by UPI. Money back after first month.",
    "Email your resume to {mail} and receive instant confirmation.",
    "Work only on your mobile phone. Unlimited income. Daily payment guaranteed.",
]
S_DESC_SUBTLE = [
    "We are hiring {t} for an international client. Flexible hours and remote work.",
    "Candidates must have a laptop and a stable internet connection. Fast hiring process.",
    "A small security deposit may be required for equipment. Salary credited weekly.",
    "Reach us at {mail} to schedule a quick chat. Positions fill fast.",
    "Responsibilities include data handling, communication with clients and daily reporting.",
]


# Offer/appointment letters previously appeared ONLY in the scam class, so the models learned
# "offer letter => scam" and flagged every genuine offer letter. Genuine offer letters must exist
# in the data for the register to carry no label information on its own.
G_OFFER = [
    "We are pleased to offer you the {t} position at {co}, starting {date}.",
    "Congratulations! Following your interview rounds we are glad to confirm this offer.",
    "You have been selected for this role after the technical and HR discussions.",
    "Your skills and experience will be a valuable asset to our team.",
    "In this full-time role you will report to the {mgr}, {mgr_title}.",
    "Your annual compensation will be {ctc}, along with benefits as per company policy.",
    "This offer is subject to the standard background verification and reference checks.",
    "Please sign and return a copy of this letter by {date} to confirm your acceptance.",
    "Your place of work will be our {city} office; working hours are 9:30 am to 6:30 pm.",
    "We look forward to welcoming you and supporting your growth at {co}.",
    "Any queries may be addressed to the HR team at hr@{dom}.",
    "Notice period during probation is 30 days, as detailed in the enclosed terms.",
    "Please find your offer letter and the detailed terms of employment enclosed with this mail.",
    "This offer letter supersedes any previous verbal discussion regarding compensation.",
]
# Fake offer letters mimic the genuine register closely and differ only in BEHAVIOUR (a fee, an
# identity-document demand, a free-mail recruiter, no interview). Keeping both classes in this
# register is what forces the model to learn the behaviour instead of the document type.
S_OFFER = [
    "Congratulations! We are pleased to offer you the {t} position at {co}, starting immediately.",
    "Your profile has been selected without any interview round.",
    "Your annual compensation will be {ctc}, credited weekly to your account.",
    "To confirm your seat and receive the offer letter, pay a refundable registration fee of {fee} on UPI.",
    "Kindly send your Aadhaar, PAN and bank account details to complete verification.",
    "Reply to {mail} only; our HR will coordinate the rest on WhatsApp.",
    "This offer is valid for 24 hours and limited seats are remaining.",
    "A security deposit of {fee} is required for the company laptop, refundable after one month.",
]
SCAM_OFFER_SHARE = 0.35  # share of scam rows written as fake offer letters

MANAGERS = ["Samuel Boateng", "Priya Raghavan", "Daniel Okafor", "Meera Nair", "Thomas Lindqvist", "Ananya Bose"]
MANAGER_TITLES = ["Senior Branch Manager", "Head of Engineering", "Director of Operations",
                  "Senior Manager, Analytics", "Regional Sales Head"]
OFFER_DATES = ["10th May 2026", "1st July 2026", "15th March 2026", "2nd January 2027"]
CTC = ["USD 72,000 per annum", "Rs 9,60,000 per annum", "GBP 48,000 per annum", "Rs 14,40,000 per annum"]
GENUINE_OFFER_SHARE = 0.18  # share of genuine rows written as offer letters rather than postings


BORDERLINE = [
    "Apply immediately, limited seats for the campus drive.",
    "Our recruiter will coordinate the schedule over WhatsApp.",
    "Salary up to Rs 50,000 per month based on skills.",
    "Urgent requirement for this role, immediate joining preferred.",
    "No prior experience needed for freshers, training is provided by us.",
    "Send your resume to {mail}.",
]


def _pick(rng: random.Random, pool: list[str], k: int) -> list[str]:
    return rng.sample(pool, min(k, len(pool)))


def _fill(rng: random.Random, s: str, title: str, company: str = "") -> str:
    amt = rng.choice(["$3000", "$5000", "Rs 50,000", "₹40,000", "$2500"])
    fee = rng.choice(["Rs 1,500", "₹2,000", "$99", "Rs 3,500", "₹999"])
    mail = f"{rng.choice(['hr', 'jobs', 'careers'])}{rng.randint(10, 99)}@{rng.choice(['gmail.com', 'yahoo.com', 'outlook.com'])}"
    co = company or rng.choice(COMPANIES)
    return s.format(t=title, city=rng.choice(CITIES), amt=amt, fee=fee, mail=mail, co=co,
                    dom=co.lower().replace(" ", "") + ".com", date=rng.choice(OFFER_DATES),
                    mgr=rng.choice(MANAGERS), mgr_title=rng.choice(MANAGER_TITLES), ctc=rng.choice(CTC))


def _genuine_row(rng: random.Random) -> dict:
    """A genuine posting, or (``GENUINE_OFFER_SHARE`` of the time) a genuine offer letter."""
    title = rng.choice(GENUINE_TITLES)
    comp = rng.choice(COMPANIES)
    if rng.random() < GENUINE_OFFER_SHARE:
        # Offer-letter register in the genuine class: without these rows the models treat
        # "we are pleased to offer you ..." as a scam signal on its own.
        sents = [G_OFFER[0], *_pick(rng, G_OFFER[1:], rng.randint(4, 6))]
        desc = " ".join(_fill(rng, x, title, comp) for x in sents)
        return {
            "title": f"Offer of employment – {title}",
            "company_profile": f"<p>{comp} is a global company building products people love.</p>",
            "description": f"<p>{desc}</p>",
            "requirements": "Signed acceptance and standard background verification.",
            "benefits": ", ".join(_pick(rng, G_BEN, 3)), "fraudulent": 0,
        }
    desc = " ".join(_fill(rng, s, title, comp) for s in _pick(rng, G_DESC, rng.randint(4, 7)))
    req = f"{rng.randint(1, 8)}+ years of experience. Skills: {', '.join(_pick(rng, SKILLS, 4))}. Bachelor's degree."
    if rng.random() < 0.2:
        desc += " " + _fill(rng, rng.choice(BORDERLINE), title)
    return {
        "title": title, "company_profile": f"<p>{comp} is a global company building products people love.</p>",
        "description": f"<p>{desc}</p>", "requirements": req,
        "benefits": ", ".join(_pick(rng, G_BEN, 3)), "fraudulent": 0,
    }


def _scam_row(rng: random.Random) -> dict:
    """A scam posting, or (``SCAM_OFFER_SHARE``) a fake offer letter.

    Signal strength varies: 20% none, 40% one cue, 40% several.
    """
    if rng.random() < SCAM_OFFER_SHARE:
        title = rng.choice(GENUINE_TITLES)
        comp = rng.choice(COMPANIES)
        sents = [S_OFFER[0], *_pick(rng, S_OFFER[1:], rng.randint(2, 4))]
        return {
            "title": f"Offer of employment – {title}",
            "company_profile": "" if rng.random() < 0.5 else f"<p>{comp} is a leading growing firm.</p>",
            "description": "<p>" + " ".join(_fill(rng, x, title, comp) for x in sents) + "</p>",
            "requirements": rng.choice(["Anyone can apply.", "Signed acceptance.", ""]),
            "benefits": rng.choice(["Weekly payout", "Work from home", ""]), "fraudulent": 1,
        }
    mimic = rng.random() < 0.4  # looks like a normal posting in the non-description fields
    title = rng.choice(GENUINE_TITLES if mimic else SCAM_TITLES)
    roll = rng.random()
    n_cues = 0 if roll < 0.2 else (1 if roll < 0.6 else rng.randint(2, 3))
    cues = _pick(rng, S_DESC_OBVIOUS, n_cues)
    sents = cues + _pick(rng, S_DESC_SUBTLE + G_DESC, rng.randint(3, 5))
    rng.shuffle(sents)
    row = _genuine_row(rng) if mimic else {
        "company_profile": "" if rng.random() < 0.6 else "We are a leading growing firm.",
        "requirements": rng.choice(["Anyone can apply.", "10th pass. Basic typing.", "Laptop and internet.", ""]),
        "benefits": rng.choice(["Weekly payout", "Unlimited bonus", "", "Work from home"]),
    }
    return {"title": title, "company_profile": row["company_profile"],
            "description": " ".join(_fill(rng, x, title) for x in sents),
            "requirements": row["requirements"], "benefits": row["benefits"], "fraudulent": 1}


def make_emscad_like(n_genuine: int = 2000, n_scam: int = 300, seed: int = config.SEED) -> pd.DataFrame:
    """EMSCAD-shaped SYNTHETIC postings, with a few injected exact/near duplicates."""
    rng = random.Random(seed)
    rows = [_genuine_row(rng) for _ in range(n_genuine)] + [_scam_row(rng) for _ in range(n_scam)]
    rows += [dict(r, title=r["title"] + " ") for r in rng.sample(rows, 40)]  # near duplicates
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "job_id", range(1, len(df) + 1))
    return df


_CUSTOM_SCAM = [
    "Congratulations! You are selected for {co}. Pay Rs {fee} registration fee on UPI {ph} to get offer letter. No interview needed.",
    "Urgent hiring work from home. Send Aadhaar {aad} and PAN {pan} on WhatsApp {ph}. Salary 45000 per month guaranteed.",
    "{co} internship offer! Deposit ₹{fee} as security, certificate in 2 days. Contact {mail} only.",
    "Join Telegram group for part time job, earn Rs 5000 daily by liking videos. Share OTP to activate account. Call {ph}.",
    "HR department {co}: your resume shortlisted. Confirm by paying training fee Rs {fee}. Reply on gmail {mail}.",
]
_CUSTOM_GENUINE = [
    "{co} is hiring a Software Engineer in Bengaluru. 2+ years Python, two interview rounds, apply via our official careers page. Salary as per industry standards.",
    "Summer internship at {co}: 8 weeks, stipend Rs 25,000 per month, selection after online assessment and interview. Apply on the company website.",
    "{co} campus recruitment drive: eligible students with 7 CGPA will be shortlisted for aptitude test followed by technical and HR interviews.",
    "Walk-in for Customer Support Associate at {co}, Hyderabad. Bring your resume and ID proof. No fee is charged at any stage of hiring.",
    "{co} invites applications for Data Analyst. Skills: SQL, Excel, Tableau. Email your CV to careers@{dom}.com. Hybrid work.",
    # Genuine offer letters: the scam rows own this register otherwise, which teaches the model
    # that any offer letter is a scam.
    "Dear Candidate, we are pleased to offer you the position of Software Engineer at {co}. Your annual compensation will be Rs 9,60,000. Please sign and return this letter by 10th May to confirm acceptance. Queries: hr@{dom}.com",
    "Congratulations! Following your interviews, {co} is pleased to extend an offer for the role of Business Analyst. Start date 1st July. This offer is subject to standard background verification. No payment is required at any stage.",
    "{co} offer of employment: Relationship Manager, reporting to the Senior Branch Manager. Annual CTC Rs 7,20,000 plus benefits as per company policy. Your appointment letter is attached; contact hr@{dom}.com with questions.",
    "You have been selected for the Data Analyst position at {co} after the online assessment and two interview rounds. Formal offer letter attached. Joining date 15th March. HR: hr@{dom}.com",
]


def make_custom_indian(n_each: int = 60, seed: int = config.SEED + 1) -> pd.DataFrame:
    """SYNTHETIC Indian-style scam/genuine texts (text,label,source) for the separate test set."""
    rng = random.Random(seed)
    cos = ["TCS", "Infosys", "Wipro", "Zoho", "HCL", "Accenture India", "Flipkart"]
    rows = []
    for label, pool in ((1, _CUSTOM_SCAM), (0, _CUSTOM_GENUINE)):
        for _ in range(n_each):
            co = rng.choice(cos)
            text = rng.choice(pool).format(
                co=co, fee=rng.choice(["1,200", "2,500", "999", "4,000"]), dom=co.split()[0].lower(),
                ph=f"+91 {rng.choice('6789')}{rng.randint(10**8, 10**9 - 1)}",
                aad=f"{rng.randint(1000, 9999)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}",
                pan="ABCDE1234F", mail=f"{co.split()[0].lower()}hr{rng.randint(1, 99)}@gmail.com",
            )
            rows.append({"text": text, "label": label, "source": "SYNTHETIC"})
    rng.shuffle(rows)
    return pd.DataFrame(rows)


_BENIGN_HOSTS = [
    "www.tcs.com", "careers.google.com", "www.infosys.com", "jobs.lever.co", "www.linkedin.com",
    "www.naukri.com", "www.wipro.com", "boards.greenhouse.io", "www.indeed.com", "www.amazon.jobs",
    "www.microsoft.com", "www.flipkart.com", "en.wikipedia.org", "github.com", "www.bbc.com",
]
_BENIGN_PATHS = ["", "/careers", "/jobs/apply", "/about-us", "/careers/search?q=engineer&loc=india", "/en/jobs/12345"]
_BAD_TLDS = ["xyz", "online", "top", "site", "click", "work", "tk", "info"]
_BAD_WORDS = ["hiring", "jobs", "apply", "offer", "career", "verify", "login", "registration"]
_BRANDS = ["tcs", "infosys", "wipro", "amazon", "google", "hcl", "flipkart", "accenture"]


def make_urls(n_each: int = 1500, seed: int = config.SEED + 2) -> pd.DataFrame:
    """SYNTHETIC URLs in Kaggle ``malicious_phish`` format (url,type)."""
    rng = random.Random(seed)
    rows: set[tuple[str, str]] = set()
    while len([r for r in rows if r[1] == "benign"]) < n_each:
        host = rng.choice(_BENIGN_HOSTS)
        rows.add((f"{rng.choice(['http', 'https'])}://{host}{rng.choice(_BENIGN_PATHS)}", "benign"))
        if rng.random() < 0.7:  # long benign sites exist too
            slug = "".join(rng.choices("abcdefghij", k=rng.randint(3, 9)))
            rows.add((f"https://www.{slug}{rng.choice(['news', 'shop', 'blog'])}.{rng.choice(['com', 'org', 'in'])}/{slug}", "benign"))
    while len([r for r in rows if r[1] != "benign"]) < n_each:
        b, w, tld = rng.choice(_BRANDS), rng.choice(_BAD_WORDS), rng.choice(_BAD_TLDS)
        kind = rng.choice(["phishing", "malware", "defacement"])
        pattern = rng.randint(0, 4)
        digits = rng.randint(1, 9999)
        url = [
            f"http://{b}-{w}.{tld}/apply?id={digits}",
            f"http://{b}{w}{digits}.{tld}/login.php",
            f"http://{rng.randint(1, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}/{w}/{b}.html",
            f"https://{w}.{b}.secure-{digits}.{tld}/verify?user={digits}&token={rng.randint(10**5, 10**6)}",
            f"https://forms.gle/{''.join(rng.choices('AbCdEfGh123', k=12))}",
        ][pattern]
        rows.add((url, kind))
    df = pd.DataFrame(sorted(rows), columns=["url", "type"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
