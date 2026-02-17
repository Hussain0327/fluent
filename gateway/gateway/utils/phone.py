import re


def normalize_e164(phone: str) -> str:
    """Normalize a phone number to E.164 format.

    Strips whitespace, dashes, parens, and ensures a leading '+'.
    Assumes US numbers if no country code provided.
    """
    digits = re.sub(r"[^\d+]", "", phone.strip())
    if digits.startswith("+"):
        return digits
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return f"+{digits}"
