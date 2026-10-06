"""README diagram (light + dark). python scripts/make_diagrams.py -> docs/img/architecture-{light,dark}.svg"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
REP = ROOT / "reports"

THEMES = {
    "light": dict(bg="#ffffff", card="#f6f8fa", border="#d0d7de", ink="#1f2328", ink2="#59636e", faint="#e6e9ed",
                  a="#2a78d6", a_soft="#e3eefb", b="#eb6834", b_soft="#fde8de", c="#1baf7a", c_soft="#dcf4ea",
                  muted="#afb8c1"),
    "dark": dict(bg="#0d1117", card="#161b22", border="#30363d", ink="#e6edf3", ink2="#9198a1", faint="#21262d",
                 a="#3987e5", a_soft="#13263d", b="#d95926", b_soft="#3a1d12", c="#199e70", c_soft="#0f2e23",
                 muted="#484f58"),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self, w, h, t, label):
        self.w, self.h, self.t, self.label, self.parts = w, h, t, label, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=13, color=None, weight=400, anchor="start", mono=False, italic=False):
        self.add(f'<text x="{x}" y="{y}" font-family="{MONO if mono else FONT}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{color or self.t["ink"]}" text-anchor="{anchor}"'
                 f'{" font-style=\"italic\"" if italic else ""}>{esc(s)}</text>')

    def rect(self, x, y, w, h, fill, stroke=None, rx=10, dash=None, sw=1.5):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke or "none"}" stroke-width="{sw}"{d}/>')

    def line(self, x1, y1, x2, y2, color=None, sw=1.5, dash=None, arrow=False, marker="ar"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if arrow else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color or self.t["ink2"]}" '
                 f'stroke-width="{sw}" stroke-linecap="round"{d}{m}/>')

    def path(self, d, color=None, sw=1.5, dash=None, arrow=False, marker="ar"):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if arrow else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color or self.t["ink2"]}" stroke-width="{sw}" '
                 f'stroke-linecap="round" stroke-linejoin="round"{da}{m}/>')

    def circle(self, x, y, r, fill, stroke=None, sw=1.5):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

    def pill(self, x, y, s, fill, color, size=11.5, anchor="middle", pad=9):
        w = len(s) * size * 0.58 + 2 * pad
        x0 = x - w / 2 if anchor == "middle" else x
        self.rect(x0, y - size - 3, w, size + 10, fill, rx=(size + 10) / 2)
        self.text(x0 + w / 2, y + 1, s, size=size, color=color, weight=600, anchor="middle")

    def node(self, x, y, w, h, title, sub=None, accent=None, dash=None, mono_title=True):
        t = self.t
        fill = t[f"{accent}_soft"] if accent else t["card"]
        stroke = t[accent] if accent else t["border"]
        self.rect(x, y, w, h, fill, stroke, dash=dash)
        cy = y + h / 2 + (-4 if sub else 5)
        self.text(x + w / 2, cy, title, size=14, weight=650, anchor="middle", mono=mono_title)
        if sub:
            for i, s in enumerate(sub if isinstance(sub, list) else [sub]):
                self.text(x + w / 2, cy + 19 + i * 16, s, size=12, color=t["ink2"], anchor="middle")

    def render(self):
        t = self.t
        defs = (f'<defs>'
                f'<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{t["ink2"]}"/></marker>'
                f'<marker id="ara" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{t["a"]}"/></marker>'
                f'<marker id="arb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{t["b"]}"/></marker></defs>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img" aria-label="{esc(self.label)}">{defs}'
                f'<rect width="{self.w}" height="{self.h}" rx="14" fill="{t["bg"]}" stroke="{t["border"]}"/>'
                + "".join(self.parts) + "</svg>")


def header(s, title, sub):
    s.text(32, 42, title, size=19, weight=700)
    s.text(32, 64, sub, size=13, color=s.t["ink2"])



def arch(t):
    s = SVG(1080, 560, t, "Architecture: a LangGraph agent in FastAPI writes every step to a SQLite trace and streams it "
                          "over SSE to a TypeScript GraphQL gateway and React console; fault injection and replay "
                          "evaluation run against the same graph.")
    header(s, "One agent graph, traced, streamed, gated and replayed",
           "Every step is written to the trace before it is streamed. Replay re-runs recorded inputs against a candidate build.")
    # python service box
    s.rect(28, 92, 640, 300, t["card"], rx=12, sw=0)
    s.text(44, 114, "PYTHON  ·  FastAPI + LangGraph", size=11, weight=700, color=t["ink2"])
    xs, y, w, h = [48, 200, 352, 504], 140, 136, 76
    names = [("classify", "TF-IDF + LR router", "+ priority model"), ("retrieve", "Chroma: past tickets", "+ synthetic KB"),
             ("draft", "offline extractive", "(Groq: pluggable)"), ("gate", "auto / approve /", "escalate")]
    for x, (n, a, b) in zip(xs, names):
        s.node(x, y, w, h, n, [a, b], accent="c" if n == "gate" else ("a" if n == "classify" else None))
    for a, b in zip(xs[:-1], xs[1:]):
        s.line(a + w, y + h / 2, b - 4, y + h / 2, arrow=True)
    s.path(f"M116,{y+h+4} L116,236 L572,236 L572,{y+h+4}", t["ink2"], 1.2, dash="3 3")
    s.text(344, 231, "every node writes a step (input, output, error, latency)", size=11, color=t["ink2"], anchor="middle")
    s.node(48, 262, 280, 60, "fault injector + guards", "5 faults · validators on each node", dash="5 4", mono_title=False)
    s.node(360, 262, 280, 60, "SQLite trace", "runs + steps · reconstructable", accent="b")
    s.line(188, 258, 188, 240, t["b"], 1.5, dash="4 4", arrow=True, marker="arb")
    s.line(500, 240, 500, 258, arrow=True)
    s.node(48, 340, 592, 40, "replay harness: recorded runs × candidate build, mocked writes, paired tests", None, mono_title=False)
    # gateway + console
    s.node(712, 140, 160, 96, "GraphQL gateway", ["Node.js · TypeScript", "graphql-yoga"], accent="a")
    s.node(900, 140, 160, 96, "React console", ["live trace", "approve / reject"], accent="a")
    s.line(640, 178, 708, 178, arrow=True)
    s.text(674, 168, "SSE", size=11, color=t["ink2"], anchor="middle", mono=True)
    s.line(872, 178, 896, 178, arrow=True)
    s.text(884, 132, "subscription", size=11, color=t["ink2"], anchor="middle")
    s.path("M980,236 C980,292 760,292 700,292 L644,292", t["ink2"], arrow=True)
    s.text(830, 286, "human decision → trace", size=11, color=t["ink2"], anchor="middle")
    # data
    s.node(600, 420, 460, 96, "data", ["16,335 English support tickets (CC BY-NC 4.0, likely synthetic)",
                                        "grouped split: 8,389 paraphrase families never cross train/test"], mono_title=False)
    s.path("M600,468 L344,468 L344,384", t["ink2"], dash="3 3", arrow=True)
    s.text(470, 460, "train / replay set", size=11, color=t["ink2"], anchor="middle")
    return s


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for th, tok in THEMES.items():
        (OUT / f"architecture-{th}.svg").write_text(arch(tok).render())
    print("ok")
