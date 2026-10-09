"""PII masking (problem 2.4): email addresses, US phone numbers, IPv4 addresses.

Each masker replaces every match with its PII-type token and returns the
masked text plus the number of replacements. When masking several PII types
over one document, run them in the order emails -> phones -> IPs: after
emails are masked the phone/IP patterns can no longer eat digits or dots
inside an address, and neither pattern matches the mask tokens themselves.
"""

import re

EMAIL_TOKEN = "|||EMAIL_ADDRESS|||"
PHONE_TOKEN = "|||PHONE_NUMBER|||"
IP_TOKEN = "|||IP_ADDRESS|||"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# US formats: optional +1 country code, area code in parens or bare, and any
# of space/hyphen/dot (or nothing) as the two separators. The lookarounds keep
# the match from starting or ending inside a longer digit run.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+1[-. ]?)?(?:\(\d{3}\)|\d{3})[-. ]?\d{3}[-. ]?\d{4}(?!\d)")

# Dotted quad with octet bounds 0-255; must not start or end inside a longer
# digit run, so an IP inside a URL is matched but fragments of longer numbers
# are not.
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IP_RE = re.compile(rf"(?<!\d){_OCTET}(?:\.{_OCTET}){{3}}(?!\d)")


def _mask(text: str, pattern: re.Pattern[str], token: str) -> tuple[str, int]:
    masked, count = pattern.subn(token, text)
    return masked, count


def mask_emails(text: str) -> tuple[str, int]:
    return _mask(text, _EMAIL_RE, EMAIL_TOKEN)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return _mask(text, _PHONE_RE, PHONE_TOKEN)


def mask_ips(text: str) -> tuple[str, int]:
    return _mask(text, _IP_RE, IP_TOKEN)
