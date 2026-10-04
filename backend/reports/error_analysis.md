# Error analysis

> **SYNTHETIC data:** examples below are generated text; the failure modes shown are artefacts of the generator, not real-world findings.

6 real error(s) and 4 near miss(es) shown (total errors available: 33).

## 1. Aligned (fusion, out-of-fold) – false negative

- True label: **scam** · score 0.17 (threshold 0.36)
- Rule flags: BRAND_IMPERSONATION, NO_HTTPS
- Text: “(no text; image/URL only)”
- Likely cause: Missed scam: rule flags fired (BRAND_IMPERSONATION, NO_HTTPS) but the model score stayed low.

## 2. Indian custom – false positive

- True label: **genuine** · score 0.72 (threshold 0.34)
- Rule flags: none
- Text: “Summer internship at Wipro: 8 weeks, stipend Rs 25,000 per month, selection after online assessment and interview. Apply on the company website.”
- Likely cause: False alarm: the model over-weights generic job vocabulary.

## 3. Indian custom – false positive

- True label: **genuine** · score 0.72 (threshold 0.34)
- Rule flags: none
- Text: “Summer internship at Infosys: 8 weeks, stipend Rs 25,000 per month, selection after online assessment and interview. Apply on the company website.”
- Likely cause: False alarm: the model over-weights generic job vocabulary.

## 4. Indian custom – false positive

- True label: **genuine** · score 0.72 (threshold 0.34)
- Rule flags: none
- Text: “Summer internship at HCL: 8 weeks, stipend Rs 25,000 per month, selection after online assessment and interview. Apply on the company website.”
- Likely cause: False alarm: the model over-weights generic job vocabulary.

## 5. Indian custom – false positive

- True label: **genuine** · score 0.72 (threshold 0.34)
- Rule flags: none
- Text: “Summer internship at Zoho: 8 weeks, stipend Rs 25,000 per month, selection after online assessment and interview. Apply on the company website.”
- Likely cause: False alarm: the model over-weights generic job vocabulary.

## 6. EMSCAD test – false positive

- True label: **genuine** · score 0.41 (threshold 0.34)
- Rule flags: PAYMENT_REQUEST, FREE_EMAIL_HR, URGENCY_MILD
- Text: “UX Designer Helios Systems is a global company building products people love. You will collaborate with cross-functional teams to deliver high quality results. This is a full time position based in our Chennai office with hybrid flexibility. We are looking for”
- Likely cause: False alarm: rules fired on wording that can also appear in genuine postings (PAYMENT_REQUEST, FREE_EMAIL_HR, URGENCY_MILD).

## 7. Aligned (fusion, out-of-fold) – near miss

- True label: **scam** · score 0.36 (threshold 0.36)
- Rule flags: BRAND_IMPERSONATION, FREE_EMAIL_HR, MESSENGER_ONLY_CONTACT, NO_HTTPS
- Text: “(no text; image/URL only)”
- Likely cause: Correct but low margin (no hard errors available) – the decision here is close to the threshold.

## 8. Aligned (fusion, out-of-fold) – near miss

- True label: **scam** · score 0.43 (threshold 0.36)
- Rule flags: BRAND_IMPERSONATION, NO_HTTPS, SUSPICIOUS_TLD
- Text: “(no text; image/URL only)”
- Likely cause: Correct but low margin (no hard errors available) – the decision here is close to the threshold.

## 9. Aligned (fusion, out-of-fold) – near miss

- True label: **scam** · score 0.43 (threshold 0.36)
- Rule flags: BRAND_IMPERSONATION, NO_HTTPS, SUSPICIOUS_TLD
- Text: “(no text; image/URL only)”
- Likely cause: Correct but low margin (no hard errors available) – the decision here is close to the threshold.

## 10. Aligned (fusion, out-of-fold) – near miss

- True label: **genuine** · score 0.27 (threshold 0.36)
- Rule flags: none
- Text: “(no text; image/URL only)”
- Likely cause: Correct but low margin (no hard errors available) – the decision here is close to the threshold.
