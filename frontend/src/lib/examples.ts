/**
 * Demo inputs. The first two are the pair that exposed the offer-letter confound: identical
 * company, role and salary, differing only in behaviour (a fee, an ID demand, a free-mail
 * recruiter). A model that classifies document type rather than legitimacy scores them the same.
 */
export interface Example {
  id: string
  label: string
  blurb: string
  expect: 'Safe' | 'Suspicious' | 'Likely Scam'
  text: string
  url?: string
}

export const EXAMPLES: Example[] = [
  {
    id: 'genuine-offer',
    label: 'Genuine offer letter',
    blurb: 'A real offer — no fee, no ID demand, corporate contact.',
    expect: 'Safe',
    text:
      'We are pleased to offer you the Relationship Manager position at Wealth Bank, starting 10th May 2026. ' +
      'Your skills and experience will be a valuable asset to our team.\n' +
      'In this full-time role, you will develop client relationships, identify financial opportunities, and ' +
      'deliver quality banking services, reporting to Samuel Boateng, Senior Branch Manager. Your annual ' +
      'compensation will be USD 72,000, along with benefits per our policies.\n' +
      'We look forward to welcoming you and supporting your growth at Wealth Bank.',
  },
  {
    id: 'fake-offer',
    label: 'Fake offer letter',
    blurb: 'Same company and role — but asks for a fee and your Aadhaar.',
    expect: 'Likely Scam',
    text:
      'Congratulations! You are selected as Relationship Manager at Wealth Bank. No interview required. ' +
      'To confirm your seat and receive the offer letter, pay a refundable registration fee of Rs 1,500 on UPI. ' +
      'Kindly send your Aadhaar and PAN details to complete verification. ' +
      'Contact HR on WhatsApp only or reply to wealthbankhr22@gmail.com',
  },
  {
    id: 'scam-forward',
    label: 'Scam WhatsApp forward',
    blurb: 'Work-from-home bait with guaranteed daily income.',
    expect: 'Likely Scam',
    text:
      'Urgent hiring work from home! Earn Rs 5,000 daily by liking videos, only 2 hours a day. ' +
      'No experience needed, selection guaranteed without interview. Limited seats, apply immediately. ' +
      'Join our Telegram group and share the OTP to activate your account.',
    url: 'http://wealthbank-hiring.online/register?id=42',
  },
  {
    id: 'genuine-posting',
    label: 'Genuine job posting',
    blurb: 'An ordinary listing with a real careers link.',
    expect: 'Safe',
    text:
      'Wealth Bank is hiring a Relationship Manager in Chennai. We are looking for a motivated candidate to ' +
      'join our growing team. Responsibilities include planning, execution and reporting of key deliverables. ' +
      'Two interview rounds followed by an HR discussion. Competitive salary as per industry standards and ' +
      'performance based bonus. 3+ years of experience. Skills: communication, Excel, SQL.',
    url: 'https://careers.wealthbank.com/apply',
  },
]
