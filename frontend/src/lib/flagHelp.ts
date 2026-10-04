import type { RedFlag } from '../api/types'

export interface FlagHelp {
  title: string
  what: string
  why: string
  action: string
}

const MODEL_WORDS: FlagHelp = {
  title: 'Words that influenced the AI score',
  what: 'Our language model compared your text with thousands of real postings. The words listed are the ones that pushed its score up the most.',
  why: 'Some words show up more often in scam postings than in genuine ones. This is only a statistical hint — a single word never proves a scam, and these words can appear in genuine postings too.',
  action: 'Use it as a pointer to re-read those parts of the message carefully. Rely on the specific warnings above (fees, mismatches, look-alike links) for the verdict.',
}

const MODEL_URL: FlagHelp = {
  title: 'Features of the web address that influenced the AI score',
  what: 'Our model looks only at how the address is written (length, digits, hyphens, ending, keywords) — it never opens the page.',
  why: 'Scam links often share patterns: long names stuffed with words like “hiring” or “offer”, many hyphens or digits, or unusual endings such as .xyz or .online.',
  action: 'Ignore the link in the message. Search for the company yourself and use the address on its official website.',
}

const GENERIC: FlagHelp = {
  title: 'Warning sign',
  what: 'Something in what you submitted matched a pattern we see in scams.',
  why: 'On its own it may be harmless, but together with other signs it raises the risk.',
  action: 'Verify the employer through its official website or HR phone number before sharing anything or paying.',
}

const BY_CODE: Record<string, FlagHelp> = {
  PAYMENT_REQUEST: {
    title: 'You are asked to pay money',
    what: 'The text asks for a registration, training, security, “refundable” or similar fee, or mentions paying by UPI or wallet.',
    why: 'Genuine employers and internship programmes do not charge candidates to hire them. Fee requests are the most common job-scam trick; the money is never refunded.',
    action: 'Do not pay anything. Stop replying, keep screenshots, and report it at cybercrime.gov.in or call 1930.',
  },
  NO_INTERVIEW: {
    title: 'Selection is promised without an interview',
    what: 'The posting says you are selected, guaranteed a job, or that no interview is needed.',
    why: 'Real hiring involves screening or interviews. “Guaranteed selection” is used to make you trust the sender quickly before asking for money or documents.',
    action: 'Ask for the interview process in writing and confirm it on the company’s official careers page.',
  },
  SENSITIVE_INFO_REQUEST: {
    title: 'Sensitive documents requested too early',
    what: 'The message asks for Aadhaar, PAN, bank details or similar documents up front.',
    why: 'Scammers use these for identity theft and bank fraud. Genuine employers ask for them only after a verified offer, through an official HR process.',
    action: 'Do not send ID or bank details. If you already did, contact your bank and report at cybercrime.gov.in / 1930.',
  },
  OTP_REQUEST: {
    title: 'You are asked for an OTP, PIN or password',
    what: 'The message asks you to share an OTP, CVV, PIN or password.',
    why: 'Nobody legitimate ever needs these. Sharing an OTP lets the sender move money out of your account or take over your accounts.',
    action: 'Never share it. If you did, call your bank immediately and block your cards.',
  },
  URGENCY: {
    title: 'Pressure to act quickly',
    what: 'The text uses urgent wording such as “limited seats”, “offer valid for 24 hours” or “apply immediately”.',
    why: 'Scammers rush you so you do not stop to verify. Genuine employers give reasonable time.',
    action: 'Take your time. Verify the company first; a real opportunity will still be there tomorrow.',
  },
  URGENCY_MILD: {
    title: 'Mentions immediate joining',
    what: 'The text mentions immediate joining or acting immediately.',
    why: 'This is common in genuine postings too, so it is only a minor signal.',
    action: 'No action needed on its own; consider it together with the other findings.',
  },
  UNREALISTIC_PAY: {
    title: 'Unrealistic earnings promised',
    what: 'The text promises very high or guaranteed pay (for example thousands per day) or “unlimited income”.',
    why: 'Pay that is far above the market for the work described is bait. Real salaries are not “guaranteed daily”.',
    action: 'Compare the pay with similar roles on trusted job sites. If it seems too good to be true, it usually is.',
  },
  MESSENGER_ONLY_CONTACT: {
    title: 'Recruitment only through WhatsApp / Telegram',
    what: 'You are asked to contact or apply through WhatsApp, Telegram or a similar chat group.',
    why: 'Scam recruiters prefer chat apps because they are hard to trace. Genuine companies use official e-mail addresses and portals.',
    action: 'Ask for a company e-mail address and confirm the contact on the official website.',
  },
  FREE_EMAIL_HR: {
    title: 'Recruiter uses a free e-mail address',
    what: 'The contact e-mail is from a free provider such as Gmail or Yahoo instead of a company domain.',
    why: 'Most companies send recruitment mail from their own domain (for example careers@company.com). Free addresses cost scammers nothing and are easy to discard.',
    action: 'Find the company’s official HR contact on its website and write there to confirm the opening.',
  },
  BRAND_IMPERSONATION: {
    title: 'Web address imitates a real company',
    what: 'The domain contains or closely resembles a well-known employer’s name (for example “tcs-hiring.online”) but is not that company’s real website.',
    why: 'Look-alike domains are cheap to register and are used to make fake application pages look official.',
    action: 'Do not use the link. Type the company’s official address yourself and look for the opening there.',
  },
  BRAND_IN_PATH: {
    title: 'Company name appears only in the link path',
    what: 'A known company’s name appears after the domain (in the path), but the site itself is not that company.',
    why: 'This can be a normal job-board page about the company, so it is a minor signal.',
    action: 'Check who owns the site and whether the opening also exists on the company’s own website.',
  },
  PLACEHOLDER_DOMAIN: {
    title: 'Link uses a placeholder address',
    what: 'The domain ends in a reserved name such as .example, .test, .invalid or .localhost, which is set aside for documentation and testing and can never belong to a real website.',
    why: 'A genuine employer links to a domain that actually exists. A reserved address means the link was copied from a template, invented, or deliberately left unusable.',
    action: 'Treat the posting as unverified. Ask the recruiter for the opening on the company’s real careers site.',
  },
  UNRECOGNIZED_EMPLOYER_DOMAIN: {
    title: 'Employer domain could not be recognised',
    what: 'The domain does not match any official company site we know of, nor a recognised job platform such as LinkedIn, Naukri or Greenhouse.',
    why: 'Many small and new employers are simply not in our list, so this on its own is a weak signal. It matters when it appears together with other warnings.',
    action: 'Search for the company independently and check the opening is listed on the site you find that way.',
  },
  IP_ADDRESS_HOST: {
    title: 'Link points to a numeric address',
    what: 'The link uses an IP address (like 192.0.2.10) instead of a company domain name.',
    why: 'Real companies use their brand domain. IP-address links are typical of throw-away scam sites.',
    action: 'Do not open it. Find the company’s official site yourself.',
  },
  URL_SHORTENER: {
    title: 'Shortened link hides the destination',
    what: 'The link uses a shortening service (bit.ly and similar).',
    why: 'You cannot see where it leads, which scammers exploit. We looked up the final destination without opening it where possible.',
    action: 'Ask for the full link, or search for the company’s careers page directly.',
  },
  FORM_OR_MESSENGER_LINK: {
    title: 'Application through a generic form or chat link',
    what: 'The application link is a free online form (such as forms.gle / Google Forms) or a chat link (t.me, wa.me) rather than a page on the company’s own website.',
    why: 'Anyone can create such a form in minutes, so scammers use them to collect your name, phone and documents. Some real start-ups also use forms, which is why this is a warning, not proof.',
    action: 'Confirm that the company really published this form: look for the same opening on its official site or LinkedIn page before filling anything in.',
  },
  SUSPICIOUS_TLD: {
    title: 'Unusual domain ending',
    what: 'The web address ends in an extension such as .xyz, .top, .online or .click.',
    why: 'These endings are cheap and heavily used by scam sites, though some legitimate sites use them too.',
    action: 'Treat it with extra caution and verify the company independently.',
  },
  AT_SYMBOL: {
    title: 'Misleading “@” in the link',
    what: 'The link contains an “@” sign.',
    why: 'Browsers ignore everything before the “@”, so a link can look like a trusted site but actually open another.',
    action: 'Do not open it.',
  },
  PUNYCODE_DOMAIN: {
    title: 'Look-alike international characters',
    what: 'The domain uses special international characters (“xn--…”).',
    why: 'Attackers use letters that look identical to Latin ones to copy a real brand’s address.',
    action: 'Do not open it; go to the company’s site by typing the address yourself.',
  },
  NO_HTTPS: {
    title: 'Link is not encrypted',
    what: 'The address starts with http:// rather than https://.',
    why: 'Genuine application pages that collect personal data almost always use HTTPS. This is a minor signal on its own.',
    action: 'Never enter personal information on a page without HTTPS.',
  },
  NEW_DOMAIN: {
    title: 'Website was created very recently',
    what: 'A public registration lookup (WHOIS) shows the domain was registered only days or weeks ago.',
    why: 'Established companies have long-standing domains; scam sites are often created just before a campaign.',
    action: 'Be very cautious and confirm the company through other channels.',
  },
  SSL_INVALID: {
    title: 'Security certificate could not be verified',
    what: 'We could not confirm a valid security (TLS) certificate for the site.',
    why: 'It may be a misconfigured site or a hastily built fake.',
    action: 'Do not enter information on the site.',
  },
  MALFORMED_URL: {
    title: 'Link could not be understood',
    what: 'The text you entered is not a valid web address, so the link was not analysed.',
    why: 'Check for typos or extra spaces.',
    action: 'Paste the full address, for example https://company.com/careers.',
  },
  QR_PAYMENT_REQUEST: {
    title: 'QR code asks for a payment',
    what: 'The image contains a UPI payment QR code.',
    why: 'A job or internship letter has no reason to include a payment QR. It is used to collect “fees” from candidates.',
    action: 'Do not scan or pay. Report it at cybercrime.gov.in / 1930.',
  },
  QR_LINK: {
    title: 'The image hides a link in a QR code',
    what: 'A QR code in the image leads to a web address. We read it without opening it and checked it like any other link.',
    why: 'QR codes hide the destination, so a fake letter can send you to a scam page.',
    action: 'Check the address shown in the link analysis before scanning anything.',
  },
  NO_TEXT_FOUND: {
    title: 'No readable text in the image',
    what: 'Text recognition (OCR) found little or no text.',
    why: 'The image may be blurry, very small or mostly graphics, so it could not be checked properly.',
    action: 'Upload a clearer, higher-resolution picture or paste the text.',
  },
  UNREADABLE_IMAGE: {
    title: 'Image could not be read',
    what: 'The file is damaged or not a supported image.',
    why: 'We ignored it and used your other inputs.',
    action: 'Upload a PNG, JPG or WebP image up to 5 MB.',
  },
  COMPANY_URL_MISMATCH: {
    title: 'The company and the link do not match',
    what: 'The text or image names a known company, but the application link belongs to a different website.',
    why: 'Genuine postings link to the company’s own domain (or a well-known careers platform). Scams reuse a real company name with their own link.',
    action: 'Ignore the link and apply through the company’s official careers page.',
  },
  COMPANY_MISMATCH_TEXT_IMAGE: {
    title: 'Text and image name different companies',
    what: 'The text names one company and the image another.',
    why: 'A genuine posting stays consistent. Mixed-up details suggest copied or fabricated material.',
    action: 'Ask the sender which company is hiring and verify it officially.',
  },
  FREE_EMAIL_FOR_NAMED_COMPANY: {
    title: 'Big company name, free e-mail address',
    what: 'The posting claims to be from a known company, but the contact e-mail is a free address (Gmail, Yahoo…).',
    why: 'Large employers send recruitment mail from their own domain. This combination is a very common impersonation pattern.',
    action: 'Do not reply to that address. Find the company’s official HR contact yourself.',
  },
  HR_EMAIL_DOMAIN_MISMATCH: {
    title: 'Recruiter e-mail does not match the company',
    what: 'The e-mail domain is not the named company’s real domain.',
    why: 'It may be a look-alike domain created to impersonate the company.',
    action: 'Verify the recruiter through the company’s official website or phone number.',
  },
  QR_URL_MISMATCH: {
    title: 'QR code leads somewhere else',
    what: 'The QR code in the image points to a different website than the link you provided.',
    why: 'Fake letters often carry a QR code to a payment or phishing page while quoting a harmless-looking link.',
    action: 'Do not scan the QR code.',
  },
  SALARY_MISMATCH: {
    title: 'Pay differs between text and image',
    what: 'The salary in the text is noticeably different from the salary in the image.',
    why: 'Genuine offers are consistent. Differences suggest edited or fabricated documents.',
    action: 'Ask for the compensation in writing from an official company e-mail.',
  },
  ROLE_MISMATCH: {
    title: 'Job role differs between text and image',
    what: 'The role named in the text is different from the role in the image.',
    why: 'Inconsistent details point to copied or fabricated material.',
    action: 'Confirm the exact role with the company’s official HR.',
  },
}

/** Find the explanation for a displayed reason: first by matching it to a known red flag, then by its wording. */
export function helpFor(reason: string, flags: RedFlag[]): FlagHelp {
  if (/driven mostly by the words|most influenced by the words/i.test(reason)) return MODEL_WORDS
  if (/AI model noticed|web address (?:is|has|ends|uses|does|contains|points|looks)/i.test(reason)) return MODEL_URL
  const hit = flags.find((f) => f.message && reason.includes(f.message))
  return (hit && BY_CODE[hit.code]) || GENERIC
}

export const KNOWN_FLAG_CODES = Object.keys(BY_CODE)
