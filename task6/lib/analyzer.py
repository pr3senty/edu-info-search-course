import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# nltk.download("stopwords")

class TextAnalyzer:
    def __init__(self):
        self.stemmers = {
            "english": SnowballStemmer("english"),
            "russian": SnowballStemmer("russian"),
        }
        self.stop = {
            "english": set(stopwords.words("english")),
            "russian": set(stopwords.words("russian")),
        }

    def _detect_lang(self, token: str) -> str:
        if re.search(r"[а-яё]", token):
            return "russian"
        return "english"

    def terms(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zа-яё]+", text.lower())
        out = []

        for t in tokens:
            lang = self._detect_lang(t)
            if t in self.stop[lang]:
                continue

            out.append(self.stemmers[lang].stem(t))

        return out