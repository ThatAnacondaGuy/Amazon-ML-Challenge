"""Probe: script-agnostic Brahmic -> Latin romanization using ONLY stdlib unicodedata character names.
Works for any Brahmic script (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam, ...)
because Unicode names all these scripts' letters with the same phonetic scheme."""
import unicodedata, re
from functools import lru_cache
IND_VOWELS = {"A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u", "VOCALIC R": "ri", "VOCALIC RR": "ri",
              "VOCALIC L": "li", "E": "e", "EE": "e", "AI": "ai", "O": "o", "OO": "o", "AU": "au", "SHORT E": "e", "SHORT O": "o",
              "CANDRA E": "e", "CANDRA O": "o", "SHORT A": "a"}
CONS = {"CA": "ch", "CHA": "chh", "TTA": "t", "TTHA": "th", "DDA": "d", "DDHA": "dh", "NNA": "n", "SSA": "sh", "SHA": "sh",
        "NYA": "ny", "NGA": "ng", "LLA": "l", "LLLA": "zh", "RRA": "r", "NNNA": "n", "RRRA": "r", "DDDHA": "rh", "QA": "q",
        "KHHA": "kh", "GHHA": "gh", "ZA": "z", "FA": "f", "YYA": "y", "JHA": "jh", "VA": "v", "WA": "v"}
NUKTA = {"k": "q", "kh": "kh", "g": "g", "j": "z", "ph": "f", "d": "r", "dh": "rh"}

@lru_cache(maxsize=None)
def classify(ch):
    try: nm = unicodedata.name(ch)
    except ValueError: return ("x", "")
    parts = nm.split(" ", 1)
    if len(parts) < 2: return ("x", "")
    rest = parts[1]
    if rest.startswith("LETTER CHILLU "): return ("c0", rest[14:].lower())            # Malayalam chillu: dead consonant
    if rest.startswith("LETTER "):
        L = rest[7:]
        if L in IND_VOWELS: return ("v", IND_VOWELS[L])
        if L in CONS: return ("c", CONS[L])
        if L.endswith("A") and L.isalpha(): return ("c", L[:-1].lower())
        return ("x", "")
    if rest.startswith("VOWEL SIGN "):
        L = rest[11:]; return ("m", IND_VOWELS.get(L, L.lower()[:2]))
    if rest in ("SIGN VIRAMA", "SIGN HALANTA") or rest.endswith("VIRAMA"): return ("vir", "")
    if rest in ("SIGN ANUSVARA", "SIGN CANDRABINDU"): return ("n", "n")
    if rest == "SIGN VISARGA": return ("n", "h")
    if rest == "SIGN NUKTA": return ("nuk", "")
    if rest.startswith("DIGIT "): return ("d", str(unicodedata.digit(ch)))
    if rest.startswith("AU LENGTH MARK") or rest.startswith("AI LENGTH MARK") or rest.startswith("LENGTH MARK"): return ("skip", "")
    return ("x", "")

def romanize(s):
    out = []; pend = False   # pend = previous consonant still carries inherent 'a'
    for ch in s:
        if ch in "‌‍": continue
        o = ord(ch)
        if o < 0x0900 or o > 0x0DFF:
            if pend: out.append("a"); pend = False
            out.append(ch); continue
        kind, val = classify(ch)
        if kind == "c":
            if pend: out.append("a")
            out.append(val); pend = True
        elif kind == "c0":
            if pend: out.append("a")
            out.append(val); pend = False
        elif kind == "m": out.append(val); pend = False
        elif kind == "vir": pend = False
        elif kind == "nuk":
            if out and out[-1] in NUKTA: out[-1] = NUKTA[out[-1]]
        elif kind in ("v", "n", "d"):
            if pend and kind != "n": out.append("a")
            if pend and kind == "n": out.append("a")
            out.append(val); pend = False
        else:
            if pend: out.append("a"); pend = False
    # word-final schwa deletion: inherent 'a' after last consonant is NOT appended (pend at word end)
    r = "".join(out)
    return r

def romanize_words(s):
    # schwa deletion handled by not flushing 'pend' at word boundaries: split on spaces first
    return " ".join(romanize(w) for w in s.split())

if __name__ == "__main__":
    for t in ["राम मार्केटिंग प्राइवेट लिमिटेड", "लक्ष्मी पावर प्रा. लि.", "भारत एनर्जी प्राइवेट लिमिटेड", "ಗುಡ್ ಪ್ರೊಡಕ್ಟ್ಸ್ ಎಲ್ಎಲ್‌ಪಿ",
              "ರಾಯಲ್ ಫುಡ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "ହ୍ୱାଇଟ୍ ୱାନ୍ ପ୍ରୋଡକ୍ଟସ୍ ଷ୍ଟୋର୍ସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍", "मॉडर्न फाइनेंस", "सिटी फूड्स",
              "विजय इंफोटेक प्राइवेट लिमिटेड", "यूनिवर्सल केयर बेकरी प्रा. लि.", "കേരളം", "ગુજરાત", "महाराष्ट्र"]:
        print(f"{t}  ->  {romanize_words(t)}")
