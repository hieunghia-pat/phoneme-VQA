import unicodedata
import re

def get_tone(word: str):
    tone_map = {
        '\u0300': '<`>',
        '\u0301': '</>',
        '\u0303': '<~>',
        '\u0309': '<?>',
        '\u0323': '<.>',
    }
    decomposed_word = unicodedata.normalize('NFD', word)
    tone = None
    remaining_word = ''
    for char in decomposed_word:
        if char in tone_map:
            tone = tone_map[char]
        else:
            remaining_word += char
    remaining_word = unicodedata.normalize('NFC', remaining_word)

    return tone, remaining_word
def get_onset(word: str) -> tuple[str, str]:
    onsets = ['ngh', 'tr', 'th', 'ph', 'nh', 'ng', 'kh',
              'gi', 'gh', 'ch', 'q', 'đ', 'x', 'v', 't',
              's', 'r', 'n', 'm', 'l', 'k', 'h', 'g', 'd',
              'c', 'b']

    # get the onset
    for onset in onsets:
        if word.startswith(onset):
            if onset != "q":
                word = word.removeprefix(onset)
            return onset, word

    return None, word
def get_medial(word: str) -> tuple[str, str]:
    O_MEDIAL = "o"
    U_MEDIAL = "u"

    if word.startswith("q"):
        # in Vietnamese, words starting with "q" always has "u" as the medial
        word = word.removeprefix("qu")
        return U_MEDIAL, word

    o_medial_cases = ["oa", "oă", "oe"]
    for o_medial_case in o_medial_cases:
        if word.startswith(o_medial_case):
            word = word.removeprefix("o")
            return O_MEDIAL, word

    if word.startswith("ua") or word.startswith("uô"):
        return None, word

    nucleuses = ['ê', 'y', 'ơ', 'a', 'â', 'ya']
    for nucleus in nucleuses:
        component = U_MEDIAL + nucleus
        if word.startswith(component):
            word = word.removeprefix("u")
            return U_MEDIAL, word

    return None, word

def get_nucleus(word: str) -> tuple[str, str]:
    nucleuses = ['oo', 'ươ', 'ưa', 'uô', 'ua', 'iê', 'yê',
                 'ia', 'ya', 'e', 'ê', 'u', 'ư', 'ô', 'i',
                 'y', 'o', 'ơ', 'â', 'a', 'o', 'ă']

    for nucleus in nucleuses:
        if word.startswith(nucleus):
            word = word.removeprefix(nucleus)
            return nucleus, word

    return None, word

def get_coda(word: str) -> str:
    codas = ['ng', 'nh', 'ch', 'u', 'n', 'o', 'p', 'c', 'm', 'y', 'i', 't']

    if word in codas:
        return word

    return None
def split_phoneme(word: str) -> list[str, str, str]:
    onset, word = get_onset(word)

    medial, word = get_medial(word)

    nucleus, word = get_nucleus(word)

    coda = get_coda(word)

    return onset, medial, nucleus, coda
def get_rhyme(word: str) -> str:
    tone, remaining_word = get_tone(word)
    onset, remaining_word = get_onset(remaining_word)
    medial, remaining_word = get_medial(remaining_word)
    nucleus, remaining_word = get_nucleus(remaining_word)
    coda = get_coda(remaining_word)

    ryhme = ""
    if medial:
        ryhme += medial
    if nucleus:
        ryhme += nucleus
    if coda:
        ryhme += coda

    return ryhme
def split_non_vietnamese_word(word):
    ONSET_SET = ["m", "b", "v",  "t",  "đ", "n", "x", "s", "l", "h", "r", "g", "d", "k", "q", "c",
                 "ph", "th", "nh", "tr", "ch", "kh", "gh", "gi", "ng", "ngh"]

    decomposed_character = unicodedata.normalize('NFD', word)

    onset = ""
    if decomposed_character in ONSET_SET:
        onset = decomposed_character
        return onset, "", ""
    else:
        return "", "", decomposed_character
def is_Vietnamese(word: str) -> tuple[bool, tuple]:
    tone, word = get_tone(word)
    if not re.match(r"[a-zA-Zăâđưôơê]", word):
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    # handling for special cases
    special_words_to_words = {
        "gin": "giin",     # gìn after being removed the tone
        "giêng": "giiêng", # giếng after being removed the tone
        "giêt": "giiêt",   # giết after being removed the tone
        "giêc": "giiêc",   # giếc (diếc) after being removed the tone
        "gi": "gii"        # gì after removing the tone
    }

    if word in special_words_to_words:
        word = special_words_to_words[word]

    # check the total number of nucleus in word
    vowels = ['oo', 'ươ', 'ưa', 'uô', 'ua', 'iê', 'yê',
              'ia', 'ya', 'e', 'ê', 'u', 'ư', 'ô', 'i',
              'y', 'o', 'ơ', 'â', 'a', 'o', 'ă']
    currentCharacterIsVowels = False
    previousCharacterIsVowels = word[0] in vowels
    foundVowels = 0

    for character in word[1:]:
        if character in vowels:
            currentCharacterIsVowels = True
        else:
            currentCharacterIsVowels = False

        if currentCharacterIsVowels and not previousCharacterIsVowels:
            foundVowels += 1

        # in Vietnamese, each word has only one syllable
        if foundVowels > 2:
            onset, medial, coda = split_non_vietnamese_word(word)
            return False, (onset, medial, coda)

        previousCharacterIsVowels = currentCharacterIsVowels

    # in case the word has the structure of a Vietnamese word, we check whether it satisfies the rule of phoneme combination
    onset, medial, nucleus, coda = split_phoneme(word)

    if nucleus is None:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    former_word = ""
    for component in [onset, medial, nucleus, coda]:
        if component is not None:
            former_word += component
    if former_word != word:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if onset == "k" and medial is None and nucleus not in ["i", "y", "e", "ê", "iê", "yê", "ia", "ya"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if onset == "c" and medial is None and nucleus in ["i", "y", "e", "ê", "iê", "yê", "ia", "ya"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if onset == "q" and not medial == "u":
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if onset == "gh" and medial is None and nucleus not in ["i", "e", "ê", "iê"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if onset == "g" and medial is None and nucleus in ["i", "e", "ê", "iê"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if onset == "ngh" and medial is None and nucleus not in ["i", "e", "ê", "iê", "yê", "ia", "ya"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if onset == "ng" and medial is None and nucleus in ["i", "e", "ê", "iê", "yê", "ia", "ya"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if medial == "o" and nucleus not in ["a", "ă", "e"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if medial == "u" and nucleus not in ['yê', 'ya', 'e', 'ê', 'y', 'ơ', "ô", 'a', 'â', 'ă']:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if nucleus == "oo" and coda not in ["ng", "c"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if nucleus == "ua" and coda is not None:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if nucleus == "ia" and coda is not None:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if nucleus == "ya" and coda is not None:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if nucleus in ["ua", "uô"] and coda == "ph":
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if nucleus in ["yê", "iê"] and coda is None:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if nucleus in ["ă", "â"] and coda is None:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if medial == "o" and nucleus in ["iê", "yê", "ia", "ya"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if medial is not None:
        if nucleus in ["u", "oo", "o", "ua", "uô", "ươ", "ưa", "ư"]:
            onset, medial, coda = split_non_vietnamese_word(word)
            return False, (onset, medial, coda)

        if nucleus in ["i", "e", "ê", "ia", "ya", "iê", "yê"] and coda in ["m", "ph"]:
            onset, medial, coda = split_non_vietnamese_word(word)
            return False, (onset, medial, coda)

    if coda == "o" and nucleus not in ["a", "e"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if coda == "y" and nucleus not in ["a", "â"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if coda == "i" and nucleus in ["ă", "â", "i", "e", "iê", "yê", "ia", "ya"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if coda == "nh" and nucleus not in ["a", "i", "y", "ê"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if coda == "ng" and nucleus not in ["a", "o", "ô", "u", "ư", "e", "iê", "ươ", "â", "ă", "uô", "oo"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if coda == "ch" and nucleus not in ["i", "a", "ê", "y"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if coda == "c" and nucleus in ["i", "ê", "e", "ơ"]:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    if nucleus == coda:
        onset, medial, coda = split_non_vietnamese_word(word)
        return False, (onset, medial, coda)

    rhyme = get_rhyme(word)
    return True, (onset, rhyme, tone)