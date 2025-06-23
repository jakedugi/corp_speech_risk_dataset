import re
import unicodedata

class TextCleaner:
    _PAGE_MARKER  = re.compile(r'^(?:Page \d+ of \d+)\s*$', flags=re.M)
    _DEHYPHEN     = re.compile(r'(\w+)-\s*\n\s*(\w+)')
    _INDENT       = re.compile(r'^[ \t]+', flags=re.M)
    _BLANK_LINES  = re.compile(r'\n{3,}')
    _FANCY_QUOTES = {
      '\u2018':"'", '\u2019':"'", '\u201C':'"', '\u201D':'"',
      '\u2013':'-', '\u2014':'-', '\u00A0':' ',
    }

    def clean(self, text: str) -> str:
        text = self._PAGE_MARKER.sub('', text)
        text = unicodedata.normalize('NFKC', text)
        for f, r in self._FANCY_QUOTES.items():
            text = text.replace(f, r)
        text = self._DEHYPHEN.sub(r'\1\2', text)
        text = self._INDENT.sub('', text)
        text = self._BLANK_LINES.sub('\n\n', text)
        text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        return text 