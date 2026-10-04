# 5-minute demo script

Setup (before the audience arrives): backend + frontend running (`uvicorn app.main:app` (in `backend/`) and `npm run dev`, or `docker compose up`),
browser at `http://localhost:5173`. The demo image is `backend/demo_assets/fake_offer_letter_with_qr.png`
(from `backend/`: `python -m scripts.make_demo_assets`; it is marked "DEMO SAMPLE – NOT A REAL OFFER").

> **Honest status note.** The outputs below were recorded on 2026-10-04 with the *stand-in models trained on SYNTHETIC data*.
> Cases 1–3 behave as intended; case 4 (genuine) currently comes out **Suspicious (~53%)** because the synthetic-trained text model is
> over-confident outside its generated vocabulary. After training on real EMSCAD (and adding DistilBERT) re-run all four cases and update
> the "Recorded" lines before presenting. Expected behaviour for a good model is stated in each case.

## 0:00 – Pitch (30 s)
"Fake job and internship scams cost students money and identity data. The base paper (Pillai, 2023) classifies text only.
TriGuard checks three things — the **text**, the **image** (offer letter / poster, including QR codes) and the **link** — and,
crucially, whether they **agree with each other**."

## 0:30 – Case 1: scam text (60 s)
Paste into the text box, click **Analyze**:
> Congratulations! You have been selected for the Data Analyst internship at TCS. No interview required. To confirm your seat, pay a refundable registration fee of Rs 2,500 on UPI within 24 hours. Send your Aadhaar and PAN card copy to tcs.hr.2025@gmail.com or contact HR on WhatsApp +91 98765 43210.

Point out: verdict badge + animated gauge; the **Why?** list with highlighted evidence (fee request, "no interview", Aadhaar/PAN request, WhatsApp-only, free e-mail
claiming to be TCS); the red "inputs don't match" banner (free Gmail for a named company).
*Recorded:* Likely Scam, 100%. *Expected with real models:* Likely Scam.

## 1:30 – Case 2: fake offer letter with QR (75 s)
Drag `fake_offer_letter_with_qr.png` into the image card. Explain: OCR reads the letter, the same text checks run on it, the QR code is **decoded but never opened**
and the link inside is analysed automatically.
Point out: image breakdown card, QR link shown, cross-modal flag *"names TCS but links to tcs-hiring.online"*.
*Recorded:* Likely Scam, ~75% (EasyOCR OCR output varies slightly). *Expected:* Likely Scam.

## 2:45 – Case 3: look-alike URL (60 s)
Clear the form, enter only `http://tcs-hiring.online/apply`.
Explain: lexical features + brand-impersonation check against ~50 known employers (rapidfuzz), no page is ever opened; WHOIS age / TLS are extra rule signals when available.
*Recorded:* Suspicious, ~44% with a **high-severity "imitates TCS (official site: tcs.com)"** reason. *Expected:* Suspicious or Likely Scam; the reason text is the point.
Optional: combine with case 1's text to show the fused score jumping and the mismatch banner.

## 3:45 – Case 4: genuine posting (45 s)
Text:
> Infosys is hiring Systems Engineers for its Bengaluru campus. Eligibility: B.E./B.Tech with 60% or above, strong fundamentals in programming and databases. The selection process consists of an online assessment, a technical interview and an HR interview. Compensation will be shared at the offer stage as per company policy. There is no fee at any stage of the hiring process. Apply only through the official careers portal.

URL: `https://www.infosys.com/careers`
Point out: no red flags, negation-aware rules ("no fee at any stage" does not trigger a payment flag), official domain recognised.
*Recorded:* Suspicious ~53% **(known limitation of the synthetic stand-in model, see note above)**. *Expected with real models:* Safe.

## 4:30 – Close (30 s)
Show **History** (anonymised) and **How it works**; mention privacy (images never stored, PII masked), calibrated scores,
and the evaluation in `backend/reports/RESULTS.md` (ablation: text vs URL vs image vs combinations, cross-modal checks).
Safety message: never pay; verify on the official careers page; report at cybercrime.gov.in / 1930.

## Backup plan
If the live server misbehaves: `python -m ml.pipeline`-style call in a notebook, or the screenshots from a dry run (take them during rehearsal).
