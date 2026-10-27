# Error analysis

9 real error(s) and 1 near miss(es) shown (total errors available: 59).

## 1. EMSCAD test – false negative

- True label: **scam** · score 0.00 (threshold 0.61)
- Rule flags: none
- Text: “Android Developer inFullMobile Sp. z o.o. is a mobile software development house, specializing in the client side of mobile software for clients all around the globe. We are a young company, where you can make the difference, the company structure is horizonta”
- Likely cause: Missed scam: no explicit cue (fee, urgency, off-platform contact); reads like an ordinary posting.

## 2. EMSCAD test – false negative

- True label: **scam** · score 0.00 (threshold 0.61)
- Rule flags: none
- Text: “ObjC Developer inFullMobile sp. z o.o. is a mobile software development house, specializing in the client side of mobile software for clients all around the globe. We are a young company, where you can make the difference, the company structure is horizontal s”
- Likely cause: Missed scam: no explicit cue (fee, urgency, off-platform contact); reads like an ordinary posting.

## 3. EMSCAD test – false negative

- True label: **scam** · score 0.01 (threshold 0.61)
- Rule flags: none
- Text: “Job for freelancer working in Saudi Arabia Hundreds of Freelance Jobs available every month for freelance designers, developers, writers and bloggers. Part of SaudiLancer - the most popular Freelance Blog and Community online | URL: www. Freelancer working in ”
- Likely cause: Missed scam: no explicit cue (fee, urgency, off-platform contact); reads like an ordinary posting.

## 4. EMSCAD test – false negative

- True label: **scam** · score 0.01 (threshold 0.61)
- Rule flags: none
- Text: “Big Data Engineer Position: Big Data Engineer Location: San Francisco, CA Duration: 6+ months CTH Skills: BI Hadoop- Java MapReduce- RDBMS- Job Description: Senior Business Intelligence engineer for Advanced Analytics overall theme. Guide and move forward an a”
- Likely cause: Missed scam: no explicit cue (fee, urgency, off-platform contact); reads like an ordinary posting.

## 5. Aligned (fusion, out-of-fold) – false negative

- True label: **scam** · score 0.14 (threshold 0.35)
- Rule flags: BRAND_IMPERSONATION, NO_HTTPS
- Text: “(no text; image/URL only)”
- Likely cause: Missed scam: rule flags fired (BRAND_IMPERSONATION, NO_HTTPS) but the model score stayed low.

## 6. Indian custom – false negative

- True label: **scam** · score 0.32 (threshold 0.61)
- Rule flags: PAYMENT_REQUEST
- Text: “Accenture India internship offer! Deposit ₹999 as security, certificate in 2 days. Contact [EMAIL] only.”
- Likely cause: Missed scam: rule flags fired (PAYMENT_REQUEST) but the model score stayed low.

## 7. Indian custom – false negative

- True label: **scam** · score 0.33 (threshold 0.61)
- Rule flags: PAYMENT_REQUEST
- Text: “Infosys internship offer! Deposit ₹1,200 as security, certificate in 2 days. Contact [EMAIL] only.”
- Likely cause: Missed scam: rule flags fired (PAYMENT_REQUEST) but the model score stayed low.

## 8. Indian custom – false negative

- True label: **scam** · score 0.33 (threshold 0.61)
- Rule flags: PAYMENT_REQUEST
- Text: “Zoho internship offer! Deposit ₹1,200 as security, certificate in 2 days. Contact [EMAIL] only.”
- Likely cause: Missed scam: rule flags fired (PAYMENT_REQUEST) but the model score stayed low.

## 9. Indian custom – false negative

- True label: **scam** · score 0.33 (threshold 0.61)
- Rule flags: PAYMENT_REQUEST
- Text: “Flipkart internship offer! Deposit ₹999 as security, certificate in 2 days. Contact [EMAIL] only.”
- Likely cause: Missed scam: rule flags fired (PAYMENT_REQUEST) but the model score stayed low.

## 10. Aligned (fusion, out-of-fold) – near miss

- True label: **scam** · score 0.36 (threshold 0.35)
- Rule flags: BRAND_IMPERSONATION, NO_HTTPS, SUSPICIOUS_TLD
- Text: “(no text; image/URL only)”
- Likely cause: Correct but low margin (no hard errors available) – the decision here is close to the threshold.
