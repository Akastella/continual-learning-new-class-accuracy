"""Redraw the ICLR overview in its original Comic Sans MS visual language.

The diagram uses editable draw.io network and image-card primitives.
Numbers come from the saved per-seed CIFAR-10 analysis, averaging four
noninitial tasks within each seed at four training epochs.

python scripts/make_overview.py --export
python scripts/make_overview.py --export --publication
The first command creates an SVG/PNG preview. The second also creates
the publication PDF/EPS. Dependencies are numpy, Pillow, Playwright,
draw.io desktop and pdftops. Install Playwright Chromium before export.
"""
from pathlib import Path
import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT / "figures/overview"
ANALYSIS = ROOT / "figure_inputs/results/r9_head_diagnostic/analysis.json"
INK, GRAY = "#202020", "#7A858C"
BLUE, TEAL = "#6188C1", "#61BEB6"
PALE_BLUE, PALE_TEAL = "#EFF7FF", "#F1F7F6"
RED, POSITIVE = "#BD4E43", "#377D86"
FONT = "Comic Sans MS"


def build():
    records = json.loads(ANALYSIS.read_text())["seed_units"]
    units = {(row["seed"], row["method"], row["view"]): row for row in records}
    differences = np.array([
        [units[s, "er", "all"]["Y"] - units[s, "sgd", "all"]["Y"],
         units[s, "er", "all"]["P"] - units[s, "sgd", "all"]["P"],
         units[s, "er", "local"]["Y"] - units[s, "sgd", "local"]["Y"]]
        for s in range(201, 211)
    ])
    means = differences.mean(axis=0)
    with (ROOT / "figure_inputs/current_only_old_accuracy.csv").open() as handle:
        old_results = list(csv.DictReader(handle))
    old_accuracy = np.mean([float(row["old"]) for row in old_results])
    mx = ET.Element("mxfile", host="app.diagrams.net", version="26.0.0")
    diagram = ET.SubElement(mx, "diagram", id="comparison-overview", name="Overview")
    model = ET.SubElement(diagram, "mxGraphModel", page="1", pageScale="1",
                          pageWidth="1100", pageHeight="710", grid="1", gridSize="10")
    cells = ET.SubElement(model, "root")
    ET.SubElement(cells, "mxCell", id="0")
    ET.SubElement(cells, "mxCell", id="1", parent="0")

    def cell(ident, label, x, y, w, h, fill="none", stroke="none", size=28,
             bold=False, color=INK, align="center", extra=""):
        style = (f"whiteSpace=wrap;html=1;fontFamily={FONT};fontSize={size};"
                 f"fontColor={color};align={align};verticalAlign=middle;spacing=0;"
                 f"strokeWidth=1.7;fillColor={fill};strokeColor={stroke};"
                 f"fontStyle={int(bold)};" + extra)
        node = ET.SubElement(cells, "mxCell", id=ident, value=label, style=style,
                             vertex="1", parent="1")
        ET.SubElement(node, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h),
                      **{"as": "geometry"})

    def line(ident, x1, y1, x2, y2, color=INK, arrow=True, points=(), width=1.8):
        node = ET.SubElement(cells, "mxCell", id=ident, value="", parent="1", edge="1",
                             style=f"strokeColor={color};strokeWidth={width};"
                             f"endArrow={'classic' if arrow else 'none'};endFill=1;")
        geometry = ET.SubElement(node, "mxGeometry", relative="1", **{"as": "geometry"})
        ET.SubElement(geometry, "mxPoint", x=str(x1), y=str(y1), **{"as": "sourcePoint"})
        ET.SubElement(geometry, "mxPoint", x=str(x2), y=str(y2), **{"as": "targetPoint"})
        if points:
            array = ET.SubElement(geometry, "Array", **{"as": "points"})
            for x, y in points:
                ET.SubElement(array, "mxPoint", x=str(x), y=str(y))

    def network(ident, x, y, color, w=94, h=64):
        points = [(x, y+8), (x, y+h-8), (x+w/2, y), (x+w/2, y+h/2),
                  (x+w/2, y+h), (x+w, y+15), (x+w, y+h-15)]
        links = [(a, b) for a in (0, 1) for b in (2, 3, 4)]
        links += [(a, b) for a in (2, 3, 4) for b in (5, 6)]
        for j, (first, second) in enumerate(links):
            line(f"{ident}_e{j}", *points[first], *points[second], color=GRAY,
                 arrow=False, width=1.3)
        for j, (px, py) in enumerate(points):
            cell(f"{ident}_n{j}", "", px-6, py-6, 12, 12,
                 fill="#FFFFFF" if j < 2 else color, stroke=INK, extra="ellipse;")

    def batch(ident, x, y):
        for j in range(3):
            xx, yy = x+7*(2-j), y+6*j
            cell(f"{ident}_sheet{j}", "", xx, yy, 46, 35, fill="#FFFFFF", stroke=BLUE)
            line(f"{ident}_mount{j}", xx+7, yy+27, xx+36, yy+27, color=BLUE,
                 arrow=False, points=[(xx+18, yy+12), (xx+27, yy+22)], width=1.3)
            cell(f"{ident}_sun{j}", "", xx+30, yy+7, 5, 5, fill=BLUE, extra="ellipse;")

    # A retains the reference's network -> new-task images -> network grammar.
    cell("headA", "a  Learn the next task", 18, 3, 1064, 46,
         size=33, bold=True)
    cell("panelA", "", 18, 57, 1064, 278, fill="#FFFFFF", stroke=INK,
         extra="rounded=1;arcSize=6;")
    cell("before", "Before training", 20, 72, 220, 45, size=28, bold=True)
    cell("train", "Train for four epochs", 250, 72, 370, 45, size=28, bold=True)
    cell("models", "After training", 623, 72, 210, 45, size=28, bold=True)
    cell("predict", "Predict", 860, 72, 195, 45, size=28, bold=True)
    network("history", 83, 164, TEAL)
    cell("historylabel", "SGD or ER<br>history", 30, 244, 200, 66, size=28)
    for key, label, y, color in [
        ("usual", "<b>Usual training</b><br>ER adds replay", 123, BLUE),
        ("new_only", "<b>New-task</b><br><b>data only</b>", 225, TEAL),
    ]:
        cell(f"{key}_group", "", 249, y, 370, 78,
             fill=PALE_TEAL if key == "usual" else PALE_BLUE, stroke=GRAY,
             extra="rounded=1;arcSize=10;dashed=1;dashPattern=6 4;")
        batch(f"{key}_data", 266, y+13)
        cell(f"{key}_label", label, 337, y+5, 268, 68, size=28)
        network(f"{key}_model", 682, y+6, color, w=92, h=62)
        line(f"{key}_train", 630, y+39, 667, y+39)
        line(f"{key}_predict", 787, y+39, 845, y+39)
        cell(f"{key}_outputs", "All 10 classes", 856, y+16, 202, 46, size=28)
    cell("new_only_old_accuracy", f"Old accuracy {old_accuracy:.0f}%",
         846, 289, 220, 31, size=26)
    line("usual_copy", 190, 184, 236, 162, points=[(228, 184), (228, 162)])
    line("new_copy", 190, 213, 236, 264, points=[(228, 213), (228, 264)])

    # B uses the final task's output indices, as in the ICLR reference.
    cell("headB", "b  Change the allowed outputs", 18, 351, 510, 48,
         size=30, bold=True)
    cell("panelB", "", 18, 409, 510, 285, fill="#FFFFFF", stroke=INK,
         extra="rounded=1;arcSize=7;")
    cell("example", "Final task · output indices", 30, 419, 486, 46, size=27)
    for row, y in [(0, 479), (1, 553)]:
        cell(f"choice{row}", "All 10" if row == 0 else "Current 2", 28, y, 137, 42,
             size=27, bold=True)
        line(f"readout{row}", 166, y+21, 179, y+21)
        for j in range(10):
            active = row == 0 or j >= 8
            cell(f"output{row}_{j}", str(j), 186+j*31, y, 27, 42,
                 fill=(PALE_BLUE if j >= 8 else PALE_TEAL) if active else "#FFFFFF",
                 stroke=(BLUE if j >= 8 else TEAL) if active else "#C9CDD0", size=27,
                 color=INK if active else "#6B7280", extra="dashed=1;" if not active else "")
    cell("same_model", "Same model parameters<br>Task identity selects outputs 8 and 9", 30, 620, 486, 63,
         size=26)

    # C has the same compact result rows and colored means as the reference.
    cell("headC", "c  The ranking changes", 550, 351, 532, 48,
         size=30, bold=True)
    cell("panelC", "", 550, 409, 532, 285, fill="#FFFFFF", stroke=INK,
         extra="rounded=1;arcSize=7;")
    cell("difference", "New-class accuracy: ER − SGD (pp)", 563, 419, 506, 46, size=27)
    for j, (label, value) in enumerate(zip([
        "Usual training · all 10", "New data only · all 10", "Usual model · current 2"
    ], means)):
        y = 474+j*48
        cell(f"comparison{j}", label, 570, y, 355, 43, size=27, align="left")
        cell(f"mean{j}", f"{value:+.2f}".replace("-", "−"), 933, y, 131, 43,
             size=32, bold=True, color=RED if value < 0 else POSITIVE, align="right")
    cell("scope", "10 seeds · 4 noninitial tasks<br>4 training epochs per task", 563, 620, 506, 63,
         size=26)
    ET.indent(mx)
    STEM.with_suffix(".drawio").write_bytes(ET.tostring(mx, encoding="utf-8", xml_declaration=True))
    (STEM.parent / "overview_data_profile.json").write_text(json.dumps({
        "source": "figure_inputs/results/r9_head_diagnostic/analysis.json",
        "seeds": list(range(201, 211)),
        "averaged_tasks": [1, 2, 3, 4],
        "epochs_per_task": 4,
        "comparisons": ["usual training", "new-task data only", "current-class prediction"],
        "er_minus_sgd_per_seed_pp": differences.tolist(),
        "er_minus_sgd_means_pp": means.tolist(),
    }, indent=2))
    print("ER minus SGD", means.round(6).tolist())


def export(drawio, publication):
    from playwright.sync_api import sync_playwright
    with tempfile.TemporaryDirectory(prefix="paa_comic_") as temporary:
        folder = Path(temporary)
        exported = folder / "drawing.svg"
        subprocess.run([drawio, "--export", "--format", "svg", "--border", "8", "--output", str(exported),
                        str(STEM.with_suffix(".drawio"))], check=True)
        deadline = time.monotonic()+30
        while not exported.exists() and time.monotonic() < deadline:
            time.sleep(.2)
        svg = exported.read_text(encoding="utf-8")
        view = [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
        height = 4.72*view[3]/view[2]
        svg = re.sub(r'width="[^"]+"', 'width="4.72in"', svg, count=1)
        svg = re.sub(r'height="[^"]+"', f'height="{height:.6f}in"', svg, count=1)
        STEM.with_suffix(".svg").write_text(svg, encoding="utf-8")
        html = (f'<!doctype html><meta charset="utf-8"><style>'
                f'@page{{size:4.72in {height:.6f}in;margin:0}}'
                'html,body{margin:0;padding:0;line-height:0;overflow:hidden}'
                'body>svg{position:absolute;left:0;top:0;display:block;width:100%;height:auto}'
                '</style>')+svg
        (folder / "preview.html").write_text(html, encoding="utf-8")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1888, "height": round(1888*view[3]/view[2])})
            page.goto((folder / "preview.html").as_uri())
            page.screenshot(path=str(STEM.with_suffix(".png")), full_page=True)
            if publication:
                page.pdf(path=str(STEM.with_suffix(".pdf")), prefer_css_page_size=True,
                         print_background=True)
            browser.close()
        with Image.open(STEM.with_suffix(".png")) as preview:
            preview.convert("L").save(STEM.with_name("overview_gray.png"))
        if publication:
            subprocess.run(["pdftops", "-eps", str(STEM.with_suffix(".pdf")),
                            str(STEM.with_suffix(".eps"))], check=True)
    print("Comic Sans overview rendered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--publication", action="store_true")
    parser.add_argument("--drawio", default=shutil.which("drawio")
                        or "C:/Program Files/draw.io/draw.io.exe")
    args = parser.parse_args()
    build()
    if args.export:
        export(args.drawio, args.publication)
