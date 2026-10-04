# Error analysis

> **SYNTHETIC data:** examples below are generated text; the failure modes shown are artefacts of the generator, not real-world findings.

9 real error(s) and 1 near miss(es) shown (total errors available: 49).

## 1. EMSCAD test – false negative

- True label: **scam** · score 0.07 (threshold 0.12)
- Rule flags: URGENCY_MILD
- Text: “QA Tester Northwind is a global company building products people love. Immediate joining is a plus but not mandatory. Apply now through our careers page. Responsibilities include data handling, communication with clients and daily reporting. We are looking for”
- Likely cause: Missed scam: rule flags fired (URGENCY_MILD) but the model score stayed low.

## 2. Indian custom – false positive

- True label: **genuine** · score 0.89 (threshold 0.12)
- Rule flags: none
- Text: “Summer internship at Zoho: 8 weeks, stipend Rs 25,000 per month, selection after online assessment and interview. Apply on the company website.”
- Likely cause: False alarm: the model over-weights generic job vocabulary.

## 3. Indian custom – false positive

- True label: **genuine** · score 0.89 (threshold 0.12)
- Rule flags: none
- Text: “Summer internship at Flipkart: 8 weeks, stipend Rs 25,000 per month, selection after online assessment and interview. Apply on the company website.”
- Likely cause: False alarm: the model over-weights generic job vocabulary.

## 4. Indian custom – false positive

- True label: **genuine** · score 0.89 (threshold 0.12)
- Rule flags: none
- Text: “Summer internship at Wipro: 8 weeks, stipend Rs 25,000 per month, selection after online assessment and interview. Apply on the company website.”
- Likely cause: False alarm: the model over-weights generic job vocabulary.

## 5. Indian custom – false positive

- True label: **genuine** · score 0.89 (threshold 0.12)
- Rule flags: none
- Text: “Summer internship at TCS: 8 weeks, stipend Rs 25,000 per month, selection after online assessment and interview. Apply on the company website.”
- Likely cause: False alarm: the model over-weights generic job vocabulary.

## 6. Aligned (fusion, out-of-fold) – false negative

- True label: **scam** · score 0.14 (threshold 0.35)
- Rule flags: BRAND_IMPERSONATION, NO_HTTPS
- Text: “(no text; image/URL only)”
- Likely cause: Missed scam: rule flags fired (BRAND_IMPERSONATION, NO_HTTPS) but the model score stayed low.

## 7. EMSCAD test – false positive

- True label: **genuine** · score 0.63 (threshold 0.12)
- Rule flags: PAYMENT_REQUEST, FREE_EMAIL_HR
- Text: “Business Analyst Helios Systems is a global company building products people love. We are looking for a motivated Business Analyst to join our growing team. Competitive salary as per industry standards and performance based bonus. Responsibilities include plan”
- Likely cause: False alarm: rules fired on wording that can also appear in genuine postings (PAYMENT_REQUEST, FREE_EMAIL_HR).

## 8. EMSCAD test – false positive

- True label: **genuine** · score 0.45 (threshold 0.12)
- Rule flags: PAYMENT_REQUEST, FREE_EMAIL_HR, URGENCY_MILD
- Text: “Software Engineer Helios Systems is a global company building products people love. This is a full time position based in our Delhi office with hybrid flexibility. We are looking for a motivated Software Engineer to join our growing team. Candidates will go th”
- Likely cause: False alarm: rules fired on wording that can also appear in genuine postings (PAYMENT_REQUEST, FREE_EMAIL_HR, URGENCY_MILD).

## 9. EMSCAD test – false positive

- True label: **genuine** · score 0.45 (threshold 0.12)
- Rule flags: PAYMENT_REQUEST, FREE_EMAIL_HR
- Text: “Software Engineer Aurora Retail is a global company building products people love. You will collaborate with cross-functional teams to deliver high quality results. Our team values ownership, learning and transparent communication. Competitive salary as per in”
- Likely cause: False alarm: rules fired on wording that can also appear in genuine postings (PAYMENT_REQUEST, FREE_EMAIL_HR).

## 10. EMSCAD test – near miss

- True label: **genuine** · score 0.12 (threshold 0.12)
- Rule flags: URGENCY
- Text: “UX Designer Helios Systems is a global company building products people love. You will collaborate with cross-functional teams to deliver high quality results. Our team values ownership, learning and transparent communication. This is a full time position base”
- Likely cause: Correct but low margin (no hard errors available) – the decision here is close to the threshold.
