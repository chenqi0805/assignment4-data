"""HTML text extraction (problem 2.2).

Decode raw HTML bytes (charset need not be UTF-8), strip anything that must
never contribute visible text (comments, script/style bodies), then hand the
document to resiliparse. Whitespace is left exactly as resiliparse emits it:
the provided fixture expects its native block formatting verbatim.
"""

import re

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

# Stripping is done before resiliparse rather than relying on its parser so the
# result does not depend on resiliparse's comment/script handling, which has
# varied across versions (e.g. text inside <!--...--> leaking into output).
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style\s*>", re.DOTALL | re.IGNORECASE)


def extract_text_from_html_bytes(b: bytes | None) -> str | None:
    """Extract plaintext from raw HTML bytes.

    Returns an empty string for ``None`` input. The bytes may be in any
    encoding resiliparse can detect (not just UTF-8).
    """
    if b is None:
        return ""
    html = b.decode(detect_encoding(b))
    html = _COMMENT_RE.sub("", html)
    html = _SCRIPT_RE.sub("", html)
    html = _STYLE_RE.sub("", html)
    return extract_plain_text(html)
