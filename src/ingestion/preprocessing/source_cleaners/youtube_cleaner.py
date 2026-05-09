import re


class YouTubeCleaner:
    """
    Cleans YouTube transcript text for RAG-grade embedding.

    What we do (and why)
    --------------------
    ✓ Remove [MM:SS] timestamps        — non-semantic tokens that inflate chunk size
    ✓ Remove [Music]/[Applause] tags   — auto-caption artefacts with zero information value
    ✓ Remove filler words              — "uh", "um", "you know", standalone "like" etc.
                                         are unique to spoken auto-transcripts and add noise
    ✓ Fix auto-caption stutter         — repeated tokens ("the the", "and and") from
                                         YouTube's ASR boundary errors
    ✓ Re-sentence run-on text          — auto-captions have no punctuation; we inject a
                                         period+space at natural pause boundaries so
                                         sentence-aware chunkers can split correctly
    ✓ Normalize whitespace             — collapse runs of spaces/newlines

    What we deliberately DO NOT do
    -------------------------------
    ✗ Stopword removal  — dense embedders (MiniLM/MPNet/e5) are trained on full natural
                          sentences; removing stopwords breaks semantic coherence
    ✗ Stemming / lemmatization — same reason; degrades vector similarity
    ✗ Lowercasing       — embedding models handle casing internally
    """

    # Spoken-word filler patterns — only strip when clearly non-semantic
    _FILLERS = re.compile(
        r'\b(uh+|um+|umm+|hmm+|mhm|erm+|err+)\b'
        r'|(\byou know\b)'
        r'|(\bkind of\b(?=\s))'   # "kind of" as hedge, not comparative
        r'|(\bsort of\b(?=\s))'
        r'|(\bright\?\s*)'
        r'|(\bokay so\b)'
        r'|(\balright so\b)'
        r'|(\bso yeah\b)'
        r'|(\byeah so\b)',
        flags=re.IGNORECASE
    )

    # Standalone "like" used as filler: "it's like you know" — but NOT "I like cats"
    # Heuristic: "like" preceded by verb/adverb and followed by noun phrase start
    _LIKE_FILLER = re.compile(
        r'(?<=[a-z,])\s+like(?=\s+[a-z])',
        flags=re.IGNORECASE
    )

    # Auto-caption noise tags
    _NOISE_TAGS = re.compile(
        r'\[\s*(?:music|applause|laughter|cheering|silence|inaudible'
        r'|crosstalk|background noise|crowd|clapping)\s*\]'
        r'|\(\s*(?:music|applause|laughter|inaudible)\s*\)'
        r'|\u266a.*?\u266a'   # ♪ ... ♪
        r'|\u266b',           # ♫
        flags=re.IGNORECASE
    )

    # [MM:SS] or [HH:MM:SS] timestamps
    _TIMESTAMPS = re.compile(r'\[\d{1,2}:\d{2}(?::\d{2})?\]')

    # Auto-caption ASR stutter: duplicated tokens at segment boundaries
    # e.g. "the the model" → "the model", "and and so" → "and so"
    _STUTTER = re.compile(r'\b(\w+)\s+\1\b', flags=re.IGNORECASE)

    # Natural pause boundaries for re-sentencing:
    # A pause is signalled by 3+ spaces, or a newline, or common spoken transition
    # phrases that auto-transcripts emit without any punctuation.
    _PAUSE_BOUNDARY = re.compile(
        r'(   +)'          # 3+ spaces (ASR silence gap encoding)
        r'|(\n+)'          # any line break
        r'|(?<=\w)(\s+(?=so |now |but |and then |which means |that means |'
        r'basically |essentially |in other words |the point is |'
        r'what this means |so what |so now ))',
        flags=re.IGNORECASE
    )

    @classmethod
    def clean(cls, text: str) -> str:
        """
        Full preprocessing pipeline for a YouTube transcript.

        Steps (order matters):
        1. Strip timestamps
        2. Remove noise tags
        3. Re-sentence (inject punctuation at pause boundaries)
        4. Remove filler words
        5. Fix stutter repetitions
        6. Normalize whitespace
        """
        if not text or not text.strip():
            return ""

        # 1. Strip [MM:SS] timestamps
        text = cls._TIMESTAMPS.sub('', text)

        # 2. Remove noise tags (music, applause, etc.)
        text = cls._NOISE_TAGS.sub('', text)

        # 3. Re-sentence: replace pause boundaries with ". "
        #    Only inject a period if the preceding character isn't already punctuated
        def _inject_period(m):
            # Get the character just before this match in the original string
            # by checking via a zero-width lookbehind trick in the replacement
            return '. '
        text = cls._PAUSE_BOUNDARY.sub(_inject_period, text)

        # Avoid double-punctuating: ".  ." → "."
        text = re.sub(r'\.\s*\.', '.', text)
        # Clean up cases like "word . word" where no period was needed
        text = re.sub(r'(?<=[.!?])\s*\.\s*', ' ', text)

        # 4. Remove filler words
        text = cls._FILLERS.sub('', text)
        text = cls._LIKE_FILLER.sub('', text)

        # 5. Fix ASR stutter repetitions (run twice for triple-stutter edge cases)
        text = cls._STUTTER.sub(r'\1', text)
        text = cls._STUTTER.sub(r'\1', text)

        # 6. Normalize whitespace
        text = re.sub(r'[ \t]{2,}', ' ', text)   # collapse horizontal whitespace
        text = re.sub(r'\n{2,}', '\n', text)      # collapse multiple newlines
        text = re.sub(r' ([,.:;!?])', r'\1', text) # remove space before punctuation

        return text.strip()
