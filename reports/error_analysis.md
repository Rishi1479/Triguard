# Error analysis

> **SYNTHETIC data:** examples below are generated text; the failure modes shown are artefacts of the generator, not real-world findings.

7 real error(s) and 3 near miss(es) shown (total errors available: 34).

## 1. Indian custom – false positive

- True label: **genuine** · score 1.00 (threshold 0.50)
- Rule flags: none
- Text: “Accenture India invites applications for Data Analyst. Skills: SQL, Excel, Tableau. Email your CV to [EMAIL]. Hybrid work.”
- Likely cause: False alarm: the model over-weights generic job vocabulary.

## 2. Indian custom – false positive

- True label: **genuine** · score 1.00 (threshold 0.50)
- Rule flags: none
- Text: “Flipkart invites applications for Data Analyst. Skills: SQL, Excel, Tableau. Email your CV to [EMAIL]. Hybrid work.”
- Likely cause: False alarm: the model over-weights generic job vocabulary.

## 3. Indian custom – false positive

- True label: **genuine** · score 1.00 (threshold 0.50)
- Rule flags: none
- Text: “TCS invites applications for Data Analyst. Skills: SQL, Excel, Tableau. Email your CV to [EMAIL]. Hybrid work.”
- Likely cause: False alarm: the model over-weights generic job vocabulary.

## 4. Indian custom – false positive

- True label: **genuine** · score 1.00 (threshold 0.50)
- Rule flags: none
- Text: “Infosys invites applications for Data Analyst. Skills: SQL, Excel, Tableau. Email your CV to [EMAIL]. Hybrid work.”
- Likely cause: False alarm: the model over-weights generic job vocabulary.

## 5. Aligned (fusion, out-of-fold) – false negative

- True label: **scam** · score 0.26 (threshold 0.50)
- Rule flags: BRAND_IMPERSONATION, NO_HTTPS
- Text: “(no text; image/URL only)”
- Likely cause: Missed scam: rule flags fired (BRAND_IMPERSONATION, NO_HTTPS) but the model score stayed low.

## 6. Aligned (fusion, out-of-fold) – false negative

- True label: **scam** · score 0.33 (threshold 0.50)
- Rule flags: BRAND_IMPERSONATION, FREE_EMAIL_HR, MESSENGER_ONLY_CONTACT, NO_HTTPS
- Text: “(no text; image/URL only)”
- Likely cause: Missed scam: rule flags fired (BRAND_IMPERSONATION, FREE_EMAIL_HR, MESSENGER_ONLY_CONTACT, NO_HTTPS) but the model score stayed low.

## 7. EMSCAD test – false negative

- True label: **scam** · score 0.42 (threshold 0.50)
- Rule flags: NO_INTERVIEW, URGENCY
- Text: “QA Tester BlueOrbit is a global company building products people love. Selection is guaranteed, there is no interview. Joining within 24 hours. Competitive salary as per industry standards and performance based bonus. Our team values ownership, learning and tr”
- Likely cause: Missed scam: rule flags fired (NO_INTERVIEW, URGENCY) but the model score stayed low.

## 8. Aligned (fusion, out-of-fold) – near miss

- True label: **scam** · score 0.52 (threshold 0.50)
- Rule flags: BRAND_IMPERSONATION, NO_HTTPS, SUSPICIOUS_TLD
- Text: “(no text; image/URL only)”
- Likely cause: Correct but low margin (no hard errors available) – the decision here is close to the threshold.

## 9. Aligned (fusion, out-of-fold) – near miss

- True label: **scam** · score 0.52 (threshold 0.50)
- Rule flags: BRAND_IMPERSONATION, NO_HTTPS, SUSPICIOUS_TLD
- Text: “(no text; image/URL only)”
- Likely cause: Correct but low margin (no hard errors available) – the decision here is close to the threshold.

## 10. Aligned (fusion, out-of-fold) – near miss

- True label: **scam** · score 0.58 (threshold 0.50)
- Rule flags: BRAND_IMPERSONATION, SUSPICIOUS_TLD
- Text: “(no text; image/URL only)”
- Likely cause: Correct but low margin (no hard errors available) – the decision here is close to the threshold.
