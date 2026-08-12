import re


def normalize(text: str) -> str:
    """
    Lowercase, strip leading/trailing whitespace, collapse
    internal whitespace runs to single spaces, remove punctuation.
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


class SeenSet:
    def __init__(self):
        self.seen = set()

    def add_and_check(self, text: str) -> bool:
        """
        Returns True if NEW (not a dupe) and records it.
        Returns False if it's a dupe.
        """
        norm = normalize(text)
        if norm in self.seen:
            return False
        self.seen.add(norm)
        return True
