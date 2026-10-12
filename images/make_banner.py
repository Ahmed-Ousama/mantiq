"""Write the README banner, images/mantiq.svg and images/mantiq-dark.svg: python images/make_banner.py images"""
import sys
from pathlib import Path

out = Path(sys.argv[1])

THEMES = {
    "light": dict(bg="#ffffff", border="#e5e7eb", card="#f9fafb", stroke="#e5e7eb", text="#111827", muted="#6b7280",
                  faint="#d1d5db", blue="#3b82f6", violet="#8b5cf6", orange="#f59e0b", step="#111827", steptext="#ffffff",
                  ask="#eef2ff", asktext="#3730a3", chip="#ede9fe", chiptext="#5b21b6", ok="#f0fdf4", okstroke="#86efac",
                  oktext="#166534", arrow="#9ca3af", newbg="#fffbeb"),
    "dark": dict(bg="#0d1117", border="#30363d", card="#161b22", stroke="#30363d", text="#e6edf3", muted="#8b949e",
                 faint="#30363d", blue="#58a6ff", violet="#a78bfa", orange="#f0b429", step="#e6edf3", steptext="#0d1117",
                 ask="#1c2541", asktext="#a5b4fc", chip="#2a1f4d", chiptext="#c4b5fd", ok="#0f2a1a", okstroke="#2ea043",
                 oktext="#7ee787", arrow="#6e7681", newbg="#2b2111"),
}

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"


def doc_card(x, y, w, name, color, c, lines=(132, 96)):
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="46" rx="8" fill="{c["card"]}" stroke="{c["stroke"]}"/>',
        f'<rect x="{x}" y="{y}" width="6" height="46" rx="3" fill="{color}"/>',
        f'<text x="{x + 20}" y="{y + 20}" font-family="{MONO}" font-size="14" fill="{c["text"]}">{name}</text>',
    ]
    for i, length in enumerate(lines[:1]):
        parts.append(f'<rect x="{x + 20}" y="{y + 30}" width="{length}" height="6" rx="3" fill="{c["faint"]}"/>')
    return "\n".join(parts)


def cache_bar(x, y, c, segments):
    """segments: (width, color, dashed)"""
    parts = [f'<text x="{x}" y="{y - 10}" font-family="{SANS}" font-size="12" font-weight="600" letter-spacing="1.2" fill="{c["muted"]}">KV CACHE</text>']
    pos = x
    for width, color, dashed, *rest in segments:
        opacity = rest[0] if rest else 1
        if dashed:
            parts.append(f'<rect x="{pos + 1}" y="{y + 1}" width="{width - 2}" height="38" rx="6" fill="{c["newbg"]}" stroke="{color}" stroke-width="2" stroke-dasharray="6 4"/>')
        else:
            parts.append(f'<rect x="{pos}" y="{y}" width="{width}" height="40" rx="6" fill="{color}" opacity="{opacity}"/>')
            for k in range(1, width // 12):
                parts.append(f'<rect x="{pos + 12 * k - 1}" y="{y + 8}" width="2" height="24" rx="1" fill="{c["bg"]}" opacity="{0.35 * opacity}"/>')
        pos += width + 4
    return "\n".join(parts)


def step(x, y, n, label, c):
    return (f'<circle cx="{x + 14}" cy="{y - 6}" r="14" fill="{c["step"]}"/>'
            f'<text x="{x + 14}" y="{y - 1}" text-anchor="middle" font-family="{SANS}" font-size="15" font-weight="700" fill="{c["steptext"]}">{n}</text>'
            f'<text x="{x + 38}" y="{y}" font-family="{SANS}" font-size="20" font-weight="600" fill="{c["text"]}">{label}</text>')


def caption(x, y, code, note, c):
    return (f'<text x="{x}" y="{y}" font-family="{MONO}" font-size="14" fill="{c["text"]}">{code}</text>'
            f'<text x="{x}" y="{y + 24}" font-family="{SANS}" font-size="14" fill="{c["muted"]}">{note}</text>')


def arrow(x1, y1, x2, y2, c):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c["arrow"]}" stroke-width="2" stroke-linecap="round" marker-end="url(#head)"/>')


def banner(c):
    W, H = 1200, 480
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="mantiq reads the knowledge into a KV cache once, answers questions with citations, and updates the cache when documents change">',
         f'<defs><marker id="head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{c["arrow"]}"/></marker></defs>',
         f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="16" fill="{c["bg"]}" stroke="{c["border"]}"/>']

    # 1. Read the knowledge once
    x = 48
    s.append(step(x, 62, 1, "Read the knowledge once", c))
    s.append(doc_card(x, 96, 220, "delivery.md", c["blue"], c, (150,)))
    s.append(doc_card(x, 152, 220, "returns.md", c["violet"], c, (120,)))
    s.append(doc_card(x, 208, 220, "prices.md", c["orange"], c, (138,)))
    s.append(arrow(x + 110, 266, x + 110, 304, c))
    s.append(f'<text x="{x + 124}" y="{290}" font-family="{SANS}" font-size="13" fill="{c["muted"]}">one pass over all documents</text>')
    s.append(cache_bar(x, 336, c, [(96, c["blue"], False), (96, c["violet"], False), (96, c["orange"], False)]))
    s.append(caption(x, 420, "prepare_kvcache_from_loader()", "saved to disk, reused for every question", c))

    s.append(arrow(372, 220, 428, 220, c))

    # 2. Answer and cite
    x = 448
    s.append(step(x, 62, 2, "Answer and cite", c))
    s.append(f'<rect x="{x + 76}" y="96" width="228" height="42" rx="21" fill="{c["ask"]}"/>')
    s.append(f'<text x="{x + 190}" y="122" text-anchor="middle" font-family="{SANS}" font-size="15" fill="{c["asktext"]}">Can I send it back?</text>')
    s.append(f'<rect x="{x}" y="148" width="304" height="46" rx="12" fill="{c["card"]}" stroke="{c["stroke"]}"/>')
    s.append(f'<text x="{x + 16}" y="176" font-family="{SANS}" font-size="15" fill="{c["text"]}">Yes, within 14 days with the receipt.</text>')
    s.append(f'<rect x="{x}" y="204" width="304" height="100" rx="12" fill="{c["card"]}" stroke="{c["stroke"]}"/>')
    s.append(f'<text x="{x + 16}" y="228" font-family="{SANS}" font-size="12" font-weight="600" letter-spacing="1.2" fill="{c["muted"]}">SOURCE</text>')
    s.append(f'<rect x="{x + 16}" y="240" width="116" height="28" rx="14" fill="{c["chip"]}" stroke="{c["violet"]}"/>')
    s.append(f'<text x="{x + 74}" y="259" text-anchor="middle" font-family="{MONO}" font-size="13" fill="{c["chiptext"]}">returns.md</text>')
    s.append(f'<rect x="{x + 142}" y="240" width="146" height="28" rx="14" fill="{c["ok"]}" stroke="{c["okstroke"]}"/>')
    s.append(f'<path d="M {x + 156} 254 l 5 5 l 9 -10" fill="none" stroke="{c["oktext"]}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>')
    s.append(f'<text x="{x + 178}" y="259" font-family="{SANS}" font-size="13" fill="{c["oktext"]}">supported 0.98</text>')
    s.append(f'<text x="{x + 16}" y="290" font-family="{SANS}" font-size="13" font-style="italic" fill="{c["muted"]}">"Returns are accepted within 14 days..."</text>')
    s.append(f'<line x1="{x + 148}" y1="306" x2="{x + 148}" y2="334" stroke="{c["violet"]}" stroke-width="2" stroke-dasharray="3 3"/>')
    s.append(cache_bar(x, 336, c, [(96, c["blue"], False, 0.22), (96, c["violet"], False), (96, c["orange"], False, 0.22)]))
    s.append(f'<text x="{x + 148}" y="394" text-anchor="middle" font-family="{SANS}" font-size="13" font-weight="600" fill="{c["violet"]}">the answer comes from here</text>')
    s.append(caption(x, 420, "cite()", "one pass with the model's citation heads", c))

    s.append(arrow(772, 220, 828, 220, c))

    # 3. Keep it up to date
    x = 848
    s.append(step(x, 62, 3, "Keep it up to date", c))
    s.append(f'<rect x="{x}" y="96" width="304" height="70" rx="8" fill="{c["card"]}" stroke="{c["stroke"]}"/>')
    s.append(f'<rect x="{x}" y="96" width="6" height="70" rx="3" fill="{c["orange"]}"/>')
    s.append(f'<text x="{x + 20}" y="120" font-family="{MONO}" font-size="14" fill="{c["text"]}">prices.md</text>')
    s.append(f'<text x="{x + 104}" y="120" font-family="{SANS}" font-size="12" fill="{c["muted"]}">changed</text>')
    s.append(f'<text x="{x + 20}" y="150" font-family="{SANS}" font-size="15" fill="{c["text"]}">TV:</text>')
    s.append(f'<text x="{x + 52}" y="150" font-family="{SANS}" font-size="15" fill="{c["muted"]}">$499</text>')
    s.append(f'<line x1="{x + 52}" y1="145" x2="{x + 88}" y2="145" stroke="{c["muted"]}" stroke-width="1.5"/>')
    s.append(f'<text x="{x + 96}" y="150" font-family="{SANS}" font-size="15" fill="{c["muted"]}">→</text>')
    s.append(f'<text x="{x + 118}" y="150" font-family="{SANS}" font-size="15" font-weight="700" fill="{c["orange"]}">$449</text>')
    s.append(arrow(x + 250, 174, x + 250, 326, c))
    s.append(cache_bar(x, 336, c, [(96, c["blue"], False), (96, c["violet"], False), (96, c["orange"], True)]))
    s.append(f'<text x="{x + 98}" y="394" text-anchor="middle" font-family="{SANS}" font-size="13" fill="{c["muted"]}">kept</text>')
    s.append(f'<line x1="{x}" y1="384" x2="{x + 196}" y2="384" stroke="{c["faint"]}" stroke-width="1.5"/>')
    s.append(f'<text x="{x + 248}" y="394" text-anchor="middle" font-family="{SANS}" font-size="13" font-weight="600" fill="{c["orange"]}">re-read</text>')
    s.append(caption(x, 420, "update_kvcache()", "only from the first change: 0.3 s, not 11.8 s", c))

    s.append("</svg>")
    return "\n".join(s)


for name, colors in THEMES.items():
    path = out / ("mantiq.svg" if name == "light" else "mantiq-dark.svg")
    path.write_text(banner(colors) + "\n")
    print("wrote", path)
