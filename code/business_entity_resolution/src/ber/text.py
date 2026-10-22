"""Rule-based normalization: cleaning, script-agnostic romanization (stdlib unicodedata only), name/address parsing,
phonetic skeleton. No external data, no pretrained models."""
import re
import unicodedata
from functools import lru_cache

# ---------------------------------------------------------------------------------------------------------------
# Romanization of Brahmic scripts driven by Unicode character names (works for every Indic script uniformly)
# ---------------------------------------------------------------------------------------------------------------
IND_VOWELS = {"A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u", "VOCALIC R": "ri", "VOCALIC RR": "ri",
              "VOCALIC L": "li", "E": "e", "EE": "e", "AI": "ai", "O": "o", "OO": "o", "AU": "au", "SHORT E": "e",
              "SHORT O": "o", "CANDRA E": "e", "CANDRA O": "o", "SHORT A": "a"}
CONS = {"CA": "ch", "CHA": "chh", "TTA": "t", "TTHA": "th", "DDA": "d", "DDHA": "dh", "NNA": "n", "SSA": "sh", "SHA": "sh",
        "NYA": "ny", "NGA": "ng", "LLA": "l", "LLLA": "zh", "RRA": "r", "NNNA": "n", "RRRA": "r", "DDDHA": "rh", "QA": "q",
        "KHHA": "kh", "GHHA": "gh", "ZA": "z", "FA": "f", "YYA": "y", "JHA": "jh", "VA": "v", "WA": "v"}
NUKTA = {"k": "q", "g": "g", "j": "z", "ph": "f", "d": "r", "dh": "rh"}
TAMIL_VOICED = {"k": "g", "t": "d", "p": "b", "ch": "s"}
INDIC_RE = re.compile(r"[ऀ-෿]")


@lru_cache(maxsize=None)
def _classify(ch):
    try:
        nm = unicodedata.name(ch)
    except ValueError:
        return ("x", "", "")
    script, _, rest = nm.partition(" ")
    if rest.startswith("LETTER CHILLU "):
        return ("c0", rest[14:].lower(), script)
    if rest.startswith("LETTER "):
        L = rest[7:]
        if L in IND_VOWELS:
            return ("v", IND_VOWELS[L], script)
        if L in CONS:
            return ("c", CONS[L], script)
        if L.endswith("A") and L.replace(" ", "").isalpha():
            return ("c", L[:-1].lower().replace(" ", ""), script)
        return ("x", "", script)
    if rest.startswith("VOWEL SIGN "):
        L = rest[11:]
        return ("m", IND_VOWELS.get(L, L.lower()[:2]), script)
    if rest.endswith("VIRAMA") or rest == "SIGN HALANTA":
        return ("vir", "", script)
    if rest in ("SIGN ANUSVARA", "SIGN CANDRABINDU"):
        return ("n", "n", script)
    if rest == "SIGN VISARGA":
        return ("vis", "h", script)
    if rest == "SIGN NUKTA":
        return ("nuk", "", script)
    if rest.startswith("DIGIT "):
        return ("d", str(unicodedata.digit(ch)), script)
    return ("x", "", script)


def _romanize_word(w):
    out, kinds, pend = [], [], False
    for ch in w:
        if ch in "‌‍":
            continue
        o = ord(ch)
        if o < 0x0900 or o > 0x0DFF:
            if pend:
                out.append("a"); kinds.append("v"); pend = False
            out.append(ch); kinds.append("o")
            continue
        kind, val, script = _classify(ch)
        if kind == "c":
            if pend:
                out.append("a"); kinds.append("v")
            if script == "TAMIL":
                prev = kinds[-1] if kinds else None
                if out and out[-1] == "h" and kinds[-1] == "vis" and val == "p":
                    out.pop(); kinds.pop(); val = "f"
                elif val == "ch" and not out:
                    val = "s"
                elif val in TAMIL_VOICED and prev in ("v", "n", "m"):
                    val = TAMIL_VOICED[val]
            if script == "MALAYALAM" and val == "r" and out and out[-1] == "r" and kinds[-1] == "c0r":
                out[-1] = "t"; val = "t"
            out.append(val); kinds.append("c"); pend = True
        elif kind == "c0":
            if pend:
                out.append("a"); kinds.append("v")
            out.append(val); kinds.append("c"); pend = False
        elif kind == "m":
            out.append(val); kinds.append("m"); pend = False
        elif kind == "vir":
            if kinds and kinds[-1] == "c" and out[-1] == "r":
                kinds[-1] = "c0r"
            pend = False
        elif kind == "nuk":
            if out and out[-1] in NUKTA:
                out[-1] = NUKTA[out[-1]]
        elif kind == "n":
            if pend:
                out.append("a"); kinds.append("v")
            out.append("m" if script == "MALAYALAM" and ch == w[-1] else "n"); kinds.append("n"); pend = False
        elif kind in ("v", "d", "vis"):
            if pend:
                out.append("a"); kinds.append("v")
            out.append(val); kinds.append(kind); pend = False
        else:
            if pend:
                out.append("a"); kinds.append("v")
            pend = False
    if pend and len(kinds) >= 2 and kinds[-2] == "c0r":
        out.append("a")
    return "".join(out)


def romanize(s):
    if not INDIC_RE.search(s):
        return s
    return " ".join(_romanize_word(w) for w in s.split())


# ---------------------------------------------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------------------------------------------
MOJIBAKE = [("Â\u0080\u0093", "-"), ("â\u0080\u0093", "-"), ("â\u0080\u0099", "'"), ("\x1a", " ")]
CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
NULL_RE = re.compile(r"<\s*null\s*>|\bnull\b|\bnan\b|\bnone\b|\bn/a\b", re.I)
DOTTED_RE = re.compile(r"\b((?:[a-z]\s?\.\s?){2,}[a-z]?)(?=\W|$)")
LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}
LEET_RE = re.compile(r"(?<![a-z0-9])([0134578])(?=[a-z]{4,}\b)|(?<=[a-z])([0134578])(?=[a-z])")


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def base_clean(s):
    if not s:
        return ""
    for a, b in MOJIBAKE:
        s = s.replace(a, b)
    s = CTRL_RE.sub(" ", s)
    s = unicodedata.normalize("NFKC", s)
    s = NULL_RE.sub(" ", s)
    s = romanize(s)
    s = strip_accents(s).lower()
    s = s.replace("'", "").replace("’", "").replace("`", "")
    s = DOTTED_RE.sub(lambda m: re.sub(r"[\s.]", "", m.group(1)), s)
    return s


def leet_fix(s):
    return LEET_RE.sub(lambda m: LEET[m.group(1) or m.group(2)], s)


# ---------------------------------------------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------------------------------------------
NAME_ABBR = {"pvt": "private", "pra": "private", "pvtltd": "private limited", "ltd": "limited", "li": "limited",
             "co": "company", "corp": "corporation", "inc": "incorporated", "intl": "international", "lnc": "incorporated",
             "mfg": "manufacturing", "svcs": "services", "svc": "services", "bros": "brothers", "assoc": "associates",
             "mgmt": "management", "ets": "etablissements", "cie": "compagnie", "frs": "freres", "n": "and"}
LEGAL = {"private", "limited", "llp", "llc", "incorporated", "corporation", "company", "lp", "plc", "pllc", "pc",
         "sas", "sarl", "sasu", "eurl", "sa", "sci", "snc", "cie", "compagnie", "gmbh", "ltda", "the", "and", "of", "public",
         "et", "de", "des", "du", "la", "le", "les"}
URL_RE = re.compile(r"https?://|www\.|\.(com|in|net|org|co|fr|biz|info|us)\b|\bid\s*:|\bdba\b|\baka\b|\bfka\b|\bformerly\b|"
                    r"\bf/k/a\b|\bd/b/a\b|\bdoing business as\b|\bknown as\b")
DOMAIN_RE = re.compile(r"\.(com|in|net|org|co|fr|biz|info|us)\b|www\.|^[@#]|\s[@#]\w")

_TD = {}


def set_translit_dict(d):
    global _TD
    _TD = d or {}


def name_fields(raw, translit=None):
    """-> (full, core, spaceless, core_tokens, skeletons, initials, is_domain, was_nonlatin)"""
    td = _TD if translit is None else translit
    raw = raw or ""
    nonlatin = bool(INDIC_RE.search(raw))
    s = base_clean(raw)
    is_domain = bool(DOMAIN_RE.search(s))
    s = URL_RE.sub(" ", s).replace("&", " and ")
    s = leet_fix(s)
    toks = re.sub(r"[^a-z0-9]+", " ", s).split()
    if nonlatin and td:
        toks = " ".join(td.get(t, t) for t in toks).split()
    toks = " ".join(NAME_ABBR.get(t, t) for t in toks).split()
    core = [t for t in toks if t not in LEGAL] or toks
    return (" ".join(toks), " ".join(core), "".join(core), core, [skel(t) for t in core],
            "".join(t[0] for t in core if t), is_domain, nonlatin)


# ---------------------------------------------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------------------------------------------
ADDR_ABBR = {"rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
             "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "hwy": "highway", "pkwy": "parkway",
             "cir": "circle", "ter": "terrace", "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor",
             "bldg": "building", "opp": "opposite", "nr": "near", "sec": "sector", "rdg": "ridge", "trl": "trail",
             "mt": "mount", "ft": "fort", "r": "rue", "bd": "boulevard", "imp": "impasse", "all": "allee",
             "che": "chemin", "ch": "chemin", "rte": "route", "crs": "cours", "q": "quai", "pass": "passage"}
ADDR_DROP = {"no", "number", "h", "hno", "house", "door", "plot", "unit", "flat", "shop", "office", "the", "of", "and",
             "near", "opposite", "behind", "floor", "suite", "apartment", "building", "c", "o", "at", "po", "de", "du",
             "des", "la", "le", "n"}
HN_RE = re.compile(r"\b[a-z]{0,2}[-/]?\d[a-z0-9\-/]*")


def _canon_hn(t):
    t = re.sub(r"[/._]", "-", t)
    t = re.sub(r"(?<=[a-z])-(?=\d)|(?<=\d)-(?=[a-z])", "", t)
    t = re.sub(r"(?<![0-9])0+(?=\d)", "", t)
    return t.strip("-")


def addr_fields(raw):
    """-> (full, word_tokens, numbers, house_number_tokens, colocation_key)"""
    s = base_clean(raw or "")
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)
    hn = sorted({_canon_hn(m) for m in HN_RE.findall(s.replace("#", " ")) if any(ch.isdigit() for ch in m)})
    toks = [ADDR_ABBR.get(t, t) for t in re.sub(r"[^a-z0-9]+", " ", s).split()]
    nums, words = [], []
    for t in toks:
        if t.isdigit():
            nums.append(t.lstrip("0") or "0")
        elif re.fullmatch(r"\d+[a-z]", t):
            nums.append(t[:-1].lstrip("0") or "0")
        elif t not in ADDR_DROP and len(t) > 1:
            words.append(t)
    key = " ".join(sorted(set(words) | set(nums)))
    return " ".join(words), words, nums, hn, key


# ---------------------------------------------------------------------------------------------------------------
# Phonetic skeleton (romanization-aware)
# ---------------------------------------------------------------------------------------------------------------
_SK = [("chh", "c"), ("ch", "c"), ("sh", "s"), ("ph", "f"), ("bh", "v"), ("kh", "k"), ("gh", "g"), ("jh", "j"),
       ("th", "t"), ("dh", "d"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("z", "j"), ("w", "v"), ("b", "v")]


@lru_cache(maxsize=1 << 20)
def skel(t):
    if not t or t.isdigit():
        return t
    t = re.sub(r"c(?=[eiy])", "s", t)
    t = re.sub(r"g(?=[eiy])", "j", t)
    for a, b in _SK:
        t = t.replace(a, b)
    t = t.replace("c", "k")
    t = re.sub(r"(.)\1+", r"\1", t)
    return t[0] + re.sub(r"[aeiouyh]", "", t[1:])
