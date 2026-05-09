import re


class WebsiteCleaner:
    """
    Cleans website-scraped text for RAG-grade embedding.

    What we do (and why)
    --------------------
    ✓ Remove boilerplate UI strings    — cookie banners, nav menus, footers, CTAs
                                         are repeated across pages and add retrieval noise
    ✓ Deduplicate repeated lines       — nav menus / sidebars often appear 2-3x in a
                                         single scrape; duplicates inflate chunk count
    ✓ Strip lone-URL lines             — bare anchor URLs (non-prose) are non-semantic
    ✓ Decode HTML entities             — &amp; &nbsp; &lt; &gt; left by parsers
    ✓ Collapse excess whitespace       — 3+ blank lines → 1 blank line

    What we deliberately DO NOT do
    -------------------------------
    ✗ Stopword removal  — dense embedders (MiniLM/MPNet/e5) are trained on full natural
                          sentences; removing stopwords breaks semantic coherence
    ✗ Stemming / lemmatization — same reason; degrades vector similarity
    ✗ Lowercasing       — embedding models handle casing internally
    """

    # ── Boilerplate line patterns (case-insensitive, applied per logical block) ──
    _BOILERPLATE = [
        # ── Original patterns (preserved unchanged) ──
        r'(?i)(accept cookies|cookie policy|privacy policy).*',
        r'(?i)(subscribe|sign up|newsletter).*',
        r'(?i)(all rights reserved|copyright ©|\bcopyright\b.*\d{4}).*',
        r'(?i)(share this|follow us on|connect with us).*',
        r'(?i)(advertisement|sponsored content|sponsored by).*',

        # ── New: navigation / structural chrome ──
        r'(?i)^(home|about|contact|blog|services|products|pricing|faq|help|support)\s*[>|»\-]?\s*$',
        r'(?i)^breadcrumb[s]?[:\s].*',
        r'(?i)^(you are here|current page)[:\s].*',
        r'(?i)^(skip to (main )?content|skip navigation)\s*$',
        r'(?i)^(back to top|scroll to top|↑\s*top)\s*$',
        r'(?i)^(read more|continue reading|learn more|see more|show more)[.…]?\s*$',
        r'(?i)^(click here|tap here|view all|see all)[.…]?\s*$',
        r'(?i)^(tags?|categories|category|labels?)[:\s].*',
        r'(?i)^(posted (in|by|on)|published (on|by)|last updated|updated on)[:\s].*',
        r'(?i)^(comments?\s*\(\d+\)|\d+\s*comments?|leave a (comment|reply))\s*$',
        r'(?i)^(related (posts?|articles?)|you (may|might) (also )?(like|enjoy))\s*$',
        r'(?i)^(table of contents?|contents?|on this page)\s*$',

        # ── New: GDPR / legal banners ──
        r'(?i)(this (website|site) uses cookies).*',
        r'(?i)(by (continuing|using) (to use )?this (site|website)).*',
        r'(?i)(gdpr|ccpa|data protection|terms (of (use|service))|legal notice).*',

        # ── New: social / engagement boilerplate ──
        r'(?i)^(tweet|share|pin it|email|print|save|bookmark)\s*$',
        r'(?i)^\d+\s*(shares?|tweets?|likes?|views?|claps?)\s*$',
        r'(?i)(join \d+ (subscribers?|readers?|followers?)).*',

        # ── New: script/style remnants left by scrapers ──
        r'(?i)^(function |var |const |let |window\.|document\.|\$\().*',
        r'(?i)^(@media|@keyframes|\.[a-z-]+\s*\{).*',
        r'(?i)^\{["\']?[@a-z_].*',         # JSON-LD fragments
        r'(?i)^(<!--|-->|</?\w+).*',         # stray HTML tags
    ]

    # Pre-compile all patterns for efficiency
    _COMPILED = [re.compile(p) for p in _BOILERPLATE]

    # Lone URL on its own line (non-prose anchor text)
    _LONE_URL = re.compile(
        r'^\s*https?://[^\s]+\s*$',
        flags=re.MULTILINE
    )

    # HTML entity decoding map (covers the common ones left by scrapers)
    _HTML_ENTITIES = [
        ('&amp;',  '&'),
        ('&nbsp;', ' '),
        ('&lt;',   '<'),
        ('&gt;',   '>'),
        ('&quot;', '"'),
        ('&#39;',  "'"),
        ('&apos;', "'"),
        ('&mdash;','—'),
        ('&ndash;','–'),
        ('&hellip;','…'),
        ('&copy;', '©'),
        ('&reg;',  '®'),
        ('&trade;','™'),
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        """
        Full preprocessing pipeline for website-scraped text.

        Steps (order matters):
        1. Decode HTML entities
        2. Remove lone-URL lines
        3. Remove boilerplate lines
        4. Deduplicate repeated lines
        5. Collapse excess whitespace
        """
        if not text or not text.strip():
            return ""

        # 1. Decode HTML entities left by the scraper/parser
        for entity, replacement in cls._HTML_ENTITIES:
            text = text.replace(entity, replacement)

        # 2. Remove lines that are bare URLs (non-prose)
        text = cls._LONE_URL.sub('', text)

        # 3. Remove boilerplate — operate line by line so patterns
        #    don't bleed across paragraph boundaries
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append('')   # preserve paragraph breaks
                continue
            if any(pat.match(stripped) for pat in cls._COMPILED):
                continue                    # drop the boilerplate line
            cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)

        # 4. Deduplicate repeated lines
        #    Navigation menus / sidebars are often scraped 2-3x in the same document.
        #    We deduplicate exact-string matches while preserving order.
        seen = set()
        deduped = []
        for line in text.splitlines():
            key = line.strip()
            if key == '':           # always keep blank lines (paragraph breaks)
                deduped.append(line)
                continue
            if key not in seen:
                seen.add(key)
                deduped.append(line)
            # silently drop the duplicate
        text = '\n'.join(deduped)

        # 5. Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)   # max 1 blank line between paragraphs
        text = re.sub(r'[ \t]{2,}', ' ', text)    # collapse horizontal whitespace

        return text.strip()
