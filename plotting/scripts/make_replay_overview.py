"""Generate the editable training and result overview from paired CIFAR-100 means."""

import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "new_experiments/shift_current_loss/summary.json"
if not SUMMARY.is_file():
    SUMMARY = ROOT.parent / "evidence/shift_current_loss/summary.json"
OUTPUT = ROOT / "figures/replay_supervision_overview.drawio"
INK, MUTED, RULE = "#22252B", "#4F5965", "#CED5DD"
BLUE, BLUE_PALE = "#305EA1", "#EEF4FC"
TEAL, TEAL_PALE = "#276C68", "#EEF7F5"


def build():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))["tasks"]
    figure = ET.Element("mxfile", host="app.diagrams.net")
    page = ET.SubElement(figure, "diagram", id="replay-loss", name="Replay supervision")
    model = ET.SubElement(page, "mxGraphModel", page="1", pageWidth="1240",
                          pageHeight="580", grid="1", gridSize="10")
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    def cell(name, x, y, w, h, label="", size=30, align="left", bold=False,
             fill="none", stroke="none", text_color=INK, extra=""):
        style = (f"html=1;whiteSpace=wrap;fontFamily=Comic Sans MS;fontSize={size};"
                 f"fontStyle={int(bold)};align={align};verticalAlign=middle;spacing=0;"
                 f"fillColor={fill};strokeColor={stroke};strokeWidth=2;fontColor={text_color};"
                 f"{extra}")
        item = ET.SubElement(root, "mxCell", id=name, value=label, vertex="1",
                             parent="1", style=style)
        ET.SubElement(item, "mxGeometry", x=str(x), y=str(y),
                      width=str(w), height=str(h), **{"as": "geometry"})

    panel = "rounded=1;arcSize=6;"
    cell("top", 10, 10, 1220, 179, fill="#FFFFFF", stroke=RULE, extra=panel)
    cell("a", 30, 21, 1170, 47, "(a) Same images in each training update", size=36, bold=True)
    for i, (label, fill, color) in enumerate([
        ("New images<br>1–32", BLUE_PALE, BLUE),
        ("New images<br>33–64", BLUE_PALE, BLUE),
        ("Old images<br>from memory", TEAL_PALE, TEAL),
        ("Repeated images<br>1–32", BLUE_PALE, BLUE),
    ]):
        cell(f"images-{i}", 30+i*299, 79, 282, 91, label, align="center",
             bold=True, fill=fill, stroke=color, extra=panel)

    cell("bottom", 10, 207, 1220, 340, fill="#FFFFFF", stroke=RULE, extra=panel)
    cell("b", 30, 218, 1170, 48, "(b) Change which classes the loss uses", size=36, bold=True)
    cell("location-note", 30, 273, 290, 75, "All-class loss<br>on new images",
         text_color=MUTED, bold=True)
    cell("loss-note", 328, 270, 496, 41, "Classes in the loss",
         align="center", text_color=MUTED)
    cell("result-note", 846, 270, 349, 41, "New / old accuracy (%)",
         size=25, align="center", bold=True, text_color=MUTED)
    for i, name in enumerate(("1–32", "33–64", "Old", "Repeats")):
        cell(f"header-{i}", 328+i*126, 310, 118, 38, name, align="center", text_color=MUTED)
    for i, task in enumerate(("8", "10")):
        cell(f"heading-{task}", 846+i*180, 310, 169, 38, f"Task {task}", align="center", text_color=MUTED)
    for index, (method, label, losses) in enumerate([
        ("mixed_restricted", "Repeated images", (False, False, True, True)),
        ("revisit_current10", "None", (False, False, True, False)),
        ("all_seen_on_nonrepeated", "Images 33–64", (False, True, True, False)),
    ]):
        y = 355 + index * 60
        cell(f"condition-{index}", 30, y, 290, 58, label, size=29, bold=True)
        for i, all_seen in enumerate(losses):
            cell(f"loss-{index}-{i}", 328+i*126, y+4, 118, 50,
                 "All seen" if all_seen else "10 new", align="center",
                 fill=TEAL_PALE if all_seen else BLUE_PALE,
                 stroke=TEAL if all_seen else BLUE, text_color=INK, extra=panel)
        for i, task in enumerate(("8", "10")):
            scores = data[task][method]
            new, old = scores["new"]["mean"], scores["old"]["mean"]
            cell(f"result-{index}-{task}", 846+i*180, y, 169, 58,
                 f"<b>{new:.2f}</b>&nbsp;/&nbsp;{old:.2f}", size=25, align="center")
    ET.indent(figure)
    ET.ElementTree(figure).write(OUTPUT, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--drawio", default=shutil.which("drawio") or "C:/Program Files/draw.io/draw.io.exe")
    args = parser.parse_args()
    build()
    if args.export:
        from make_overview import export

        export(args.drawio, True, OUTPUT.with_suffix(""))
