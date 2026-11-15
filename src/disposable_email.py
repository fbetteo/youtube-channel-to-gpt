from pathlib import Path

_DOMAINS_FILE = Path(__file__).with_name("disposable_email_domains.txt")

DISPOSABLE_DOMAINS = frozenset(
    line.strip().lower()
    for line in _DOMAINS_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
)


def is_disposable_email(email: str | None) -> bool:
    if not email or "@" not in email:
        return False
    domain = email.rsplit("@", 1)[1].strip().lower()
    # Also match subdomains such as "x.mailinator.com".
    parts = domain.split(".")
    return any(".".join(parts[i:]) in DISPOSABLE_DOMAINS for i in range(len(parts) - 1))
