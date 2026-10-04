import pytest

from ml.common.red_flags import detect_red_flags, extract_entities


def codes(text):
    return {f.code for f in detect_red_flags(text)}


@pytest.mark.parametrize("text", [
    "Pay a refundable registration fee of Rs 1,500 to confirm your seat.",
    "Training kit charges ₹2,000 to be paid by UPI.",
    "Please deposit $99 to receive your offer letter.",
    "A security deposit is required for equipment.",
    "Send payment via Western Union to start.",
    "Transfer Rs. 3500 on Paytm for verification",
])
def test_payment_positive(text):
    assert "PAYMENT_REQUEST" in codes(text)


@pytest.mark.parametrize("text", [
    "We never charge any fee at any stage of hiring.",
    "No registration fee is required.",
    "Competitive salary of Rs 50,000 per month with provident fund.",
    "The candidate will not be asked to pay any security deposit.",
    "Our fee structure for clients is on the website.",
])
def test_payment_negative(text):
    assert "PAYMENT_REQUEST" not in codes(text)


@pytest.mark.parametrize("text", ["Urgent hiring, apply immediately!", "Limited seats, hurry.", "Offer valid for 24 hours only"])
def test_urgency_positive(text):
    assert any(c.startswith("URGENCY") for c in codes(text))


def test_mild_urgency_is_low_severity():
    flags = detect_red_flags("Immediate joining is a plus.")
    assert [f.severity for f in flags] == ["low"]
    assert detect_red_flags("We offer flexible hours and mentoring.") == []


@pytest.mark.parametrize("text", [
    "Selection is guaranteed, no interview needed.", "Direct joining without interview.", "100% job guarantee? we offer guaranteed placement",
])
def test_no_interview_positive(text):
    assert "NO_INTERVIEW" in codes(text)


def test_no_interview_negative():
    assert "NO_INTERVIEW" not in codes("Selection involves two interview rounds and a technical test.")
    assert "NO_INTERVIEW" not in codes("There is never a guaranteed job; you must clear the interview.")


@pytest.mark.parametrize("text", [
    "Earn up to $5000 per week from home.", "Earn Rs 5000 daily by liking videos.", "Unlimited income guaranteed", "Daily payment assured",
])
def test_unrealistic_pay_positive(text):
    assert "UNREALISTIC_PAY" in codes(text)


def test_unrealistic_pay_negative():
    assert "UNREALISTIC_PAY" not in codes("Salary Rs 45,000 per month, paid on the last working day.")


@pytest.mark.parametrize("text", ["Contact HR on WhatsApp only.", "Join our Telegram group to apply.", "Message us on whatsapp to apply"])
def test_messenger_positive(text):
    assert "MESSENGER_ONLY_CONTACT" in codes(text)


def test_messenger_negative():
    assert "MESSENGER_ONLY_CONTACT" not in codes("We never recruit via WhatsApp or Telegram.")
    assert "MESSENGER_ONLY_CONTACT" not in codes("Our team collaborates over chat tools.")


def test_free_email_flag():
    f = detect_red_flags("Send your resume to hr.jobs@gmail.com to apply")
    assert [x.code for x in f] == ["FREE_EMAIL_HR"] and f[0].evidence == "hr.jobs@gmail.com"
    assert "FREE_EMAIL_HR" not in codes("Apply at careers@infosys.com")
    assert "FREE_EMAIL_HR" not in codes("My friend's mail is rahul@gmail.com, we had lunch.")


@pytest.mark.parametrize("text,code", [
    ("Send your Aadhaar and PAN card copy to complete verification.", "SENSITIVE_INFO_REQUEST"),
    ("Please share your bank account details.", "SENSITIVE_INFO_REQUEST"),
    ("Share OTP to activate your account.", "OTP_REQUEST"),
    ("Provide your CVV to confirm.", "OTP_REQUEST"),
])
def test_sensitive_positive(text, code):
    assert code in codes(text)


@pytest.mark.parametrize("text", [
    "Never share your OTP or bank details with anyone.",
    "We will never ask for Aadhaar or PAN before the offer.",
    "Candidates should hold a valid passport.",
])
def test_sensitive_negative(text):
    assert not codes(text) & {"SENSITIVE_INFO_REQUEST", "OTP_REQUEST"}


def test_flag_shape_and_ordering():
    flags = detect_red_flags("Urgent! Pay Rs 2,000 fee. Contact on WhatsApp only. Email hr@gmail.com")
    sev = [f.severity for f in flags]
    assert sev == sorted(sev, key=["high", "medium", "low"].index)
    assert all(f.evidence and f.message and f.code for f in flags)


def test_empty_and_clean():
    assert detect_red_flags("") == [] and detect_red_flags("   ") == []
    assert detect_red_flags("We are hiring a Python developer. Two interview rounds. Apply on our careers page.") == []


def test_extract_entities():
    e = extract_entities(
        "Infosys is hiring Systems Engineers. Visit www.infosys.com/careers or https://forms.gle/AbC123, "
        "mail careers@infosys.com, call +91 98765 43210. Northwind Technologies also, via t.me/joinjobs.")
    assert "www.infosys.com/careers" in e["urls"] and "https://forms.gle/AbC123" in e["urls"]
    assert any(u.startswith("t.me/") for u in e["urls"])
    assert e["emails"] == ["careers@infosys.com"] and "infosys.com" not in e["urls"]
    assert e["phones"] == ["+91 98765 43210"]
    assert "Infosys" in e["companies"] and "Northwind Technologies" in e["companies"]
    assert "Systems" not in e["companies"]


def test_extract_entities_empty():
    assert extract_entities("") == {"urls": [], "emails": [], "phones": [], "companies": []}


def test_negation_does_not_leak_across_sentences():
    assert "PAYMENT_REQUEST" in codes("No interview required. Pay refundable registration fee Rs 1,500 now.")
    assert "PAYMENT_REQUEST" not in codes("Pay attention to details. We never ask for any registration fee.")
