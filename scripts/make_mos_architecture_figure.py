#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the MoS (Mixed-Start flow matching) architecture figure.

Replicates Figure 3 of the GR00T N1 paper ("GR00T N1 Model Architecture")
as an editable .drawio file, and extends it with the MoS modules:
  - Memory Encoder (Transformer, 3.1M) producing the semantic start z0
  - Mixed-Start node  s = beta*rtc(h) + (1-beta)*z0 + eta*eps
  - Confidence Head   beta_hat (dashed edge)

The drawio desktop CLI (this build) cannot rasterize embedded bitmap
images, so the camera / robot photos are placed as magenta-bordered
placeholder boxes in the .drawio and then composited onto the exported
PNG with PIL (exact placeholder bbox replacement).

Usage:
  python scripts/make_mos_architecture_figure.py \
      --out figures/mos_architecture_mixed_start.drawio
  "C:\\Program Files\\draw.io\\draw.io.exe" -x -f png -s 2 \
      -o figures/_mos_arch_raw.png figures/mos_architecture_mixed_start.drawio
  python scripts/make_mos_architecture_figure.py \
      --composite-raw figures/_mos_arch_raw.png \
      --composite-out figures/mos_architecture_mixed_start.png
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- palette ---
C_GREEN_F = "#D5E8D4"; C_GREEN_S = "#82B366"      # embodiment-specific
C_WHITE_F = "#FFFFFF"; C_WHITE_S = "#666666"      # pre-trained modules
C_GREY_F  = "#E6E6E6"; C_GREY_S  = "#999999"      # attention blocks
C_TEAL_F  = "#B5E3E0"; C_TEAL_S  = "#1D9A8F"      # MoS memory / confidence
C_YEL_F   = "#FFE6B3"; C_YEL_S   = "#D79B00"      # MoS mixed-start
C_TOK_VIS = "#A66CFF"                              # vision tokens
C_TOK_TXT = "#2D7CFF"                              # text tokens
C_TOK_ST  = "#E85454"                              # state tokens
C_TOK_ACT = "#FFE29A"; C_TOK_ACT_S = "#D6B656"     # action tokens
C_TOK_OUT = "#FF9933"                              # output tokens
C_EDGE    = "#555555"
MARK      = "#FF00FF"                              # placeholder border marker


class Diagram:
    def __init__(self):
        self.cells = []
        self.n = 0

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def box(self, x, y, w, h, label="", style=""):
        cid = self._id()
        import html as _h
        self.cells.append(
            f'<mxCell id="{cid}" value="{_h.escape(label, quote=True)}" '
            f'style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f'</mxCell>')
        return cid

    def edge(self, src, dst, style="", label="", pts=None):
        import html as _h
        cid = self._id()
        pts_xml = ""
        if pts:
            pts_xml = "<Array as='points'>" + "".join(
                f"<mxPoint x='{p[0]}' y='{p[1]}'/>" for p in pts) + "</Array>"
        self.cells.append(
            f'<mxCell id="{cid}" value="{_h.escape(label, quote=True)}" '
            f'style="{style}" edge="1" parent="1" source="{src}" target="{dst}">'
            f'<mxGeometry relative="1" as="geometry">{pts_xml}</mxGeometry>'
            f'</mxCell>')
        return cid

    def token_col(self, x, y, n, fill, stroke=None, size=16, gap=5):
        ids = []
        stroke = stroke or fill
        for i in range(n):
            ids.append(self.box(
                x, y + i * (size + gap), size, size, "",
                f"fillColor={fill};strokeColor={stroke};html=1;"))
        return ids

    def xml(self):
        body = "\n".join(self.cells)
        return (
            '<mxfile host="app.diagrams.net" type="device">'
            '<diagram name="MoS Architecture" id="mos-arch">'
            '<mxGraphModel dx="1700" dy="950" grid="0" gridSize="10" guides="1" '
            'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            'pageWidth="1750" pageHeight="960" math="0" shadow="0">'
            '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
            f'{body}</root></mxGraphModel></diagram></mxfile>')


# --------------------------------------------------------------- styles -----
S_WHITE = ("rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
           f"fillColor={C_WHITE_F};strokeColor={C_WHITE_S};fontSize=14;fontStyle=1;")
S_GREEN = ("rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
           f"fillColor={C_GREEN_F};strokeColor={C_GREEN_S};fontSize=14;fontStyle=1;")
S_TEAL  = ("rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
           f"fillColor={C_TEAL_F};strokeColor={C_TEAL_S};fontSize=13;fontStyle=1;")
S_YEL   = ("rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
           f"fillColor={C_YEL_F};strokeColor={C_YEL_S};fontSize=13;fontStyle=1;")
S_ATTN  = ("rounded=1;arcSize=8;whiteSpace=wrap;html=1;horizontal=0;"
           f"fillColor={C_GREY_F};strokeColor={C_GREY_S};fontSize=12;")
S_DASH  = ("rounded=1;arcSize=8;whiteSpace=wrap;html=1;dashed=1;dashPattern=4 3;"
           f"fillColor={C_WHITE_F};strokeColor={C_WHITE_S};fontSize=14;")
S_CONT  = ("rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=none;"
           "strokeColor=#333333;fontSize=13;")
S_TEXT  = ("text;html=1;align=center;verticalAlign=middle;fontSize=12;"
           "fontStyle=2;whiteSpace=wrap;")
S_LBL   = "text;html=1;align=center;verticalAlign=middle;fontSize=13;"
S_LBL_L = "text;html=1;align=left;verticalAlign=middle;fontSize=13;"
S_PH    = ("rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
           f"strokeColor={MARK};strokeWidth=2;fontSize=11;fontStyle=2;"
           "fontColor=#999999;")
S_EDGE  = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
           f"strokeColor={C_EDGE};strokeWidth=1.5;endArrow=open;endSize=8;")
S_EDGE_TEAL = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
               f"strokeColor={C_TEAL_S};strokeWidth=2;endArrow=open;endSize=8;")
S_EDGE_DOT  = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;dashed=1;"
               "dashPattern=2 3;strokeColor=#333333;strokeWidth=1.5;"
               "endArrow=open;endSize=8;")


def build():
    d = Diagram()

    # ====================================================== top: vision/text
    cam = d.box(40, 45, 190, 140, "camera\nimage", S_PH)
    vis_enc = d.box(280, 55, 130, 120, "Vision\nEncoder", S_WHITE)
    vis_tok = d.token_col(445, 60, 6, C_TOK_VIS)[0]

    vlm = d.box(500, 40, 140, 220, "Eagle-2\nVLM", S_WHITE)
    d.box(612, 48, 26, 26, "&#10052;",
          "text;html=1;fontSize=20;fontColor=#2D7CFF;align=center;")
    vlm_out = d.token_col(665, 45, 5, C_TOK_VIS)[0]
    vlm_out_t = d.token_col(665, 160, 4, C_TOK_TXT)[0]

    txt = d.box(40, 208, 190, 90,
                "<i>&quot;Pick up the apple and place it into the bottom shelf&quot;</i>",
                S_TEXT)
    tok = d.box(280, 215, 130, 70, "Text\nTokenizer", S_WHITE)
    txt_tok = d.token_col(445, 220, 4, C_TOK_TXT)[0]

    d.edge(cam, vis_enc, S_EDGE)
    d.edge(vis_enc, vis_tok, S_EDGE)
    d.edge(vis_tok, vlm, S_EDGE)
    d.edge(txt, tok, S_EDGE)
    d.edge(tok, txt_tok, S_EDGE)
    d.edge(txt_tok, vlm, S_EDGE)

    # ================================================= MoS: memory encoder ==
    mem_enc = d.box(500, 310, 140, 90,
                    "Memory Encoder\n(Transformer, 3.1M)\nVL-only", S_TEAL)
    z0_tok = d.token_col(665, 320, 3, "#5FC9BF", C_TEAL_S)[0]
    d.box(690, 345, 40, 22, "z₀", S_LBL_L)
    d.edge(vlm, mem_enc, S_EDGE_TEAL)
    d.edge(mem_enc, z0_tok, S_EDGE_TEAL)

    # ===================================================== middle: state ====
    st_lbl = d.box(40, 365, 200, 30, "Robot State   q<sub>t</sub>", S_LBL)
    st_enc = d.box(280, 345, 130, 80, "State\nEncoder", S_GREEN)
    st_tok = d.token_col(445, 360, 2, C_TOK_ST)[0]
    d.edge(st_lbl, st_enc, S_EDGE)
    d.edge(st_enc, st_tok, S_EDGE)

    # ===================================================== bottom: action ===
    act_lbl = d.box(40, 530, 120, 30, "Noised Action", S_LBL)
    act_box = d.box(175, 460, 90, 190,
                    "a<sub>t</sub><br><br>a<sub>t+1</sub><br><br>&#8942;<br><br>a<sub>t+H-1</sub>",
                    S_DASH)
    d.box(30, 657, 150, 22, "start = s",
          f"text;html=1;fontSize=12;fontColor={C_YEL_S};fontStyle=1;align=right;")
    act_enc = d.box(280, 480, 130, 150, "Action\nEncoder", S_GREEN)
    act_tok = d.token_col(445, 490, 8, C_TOK_ACT, C_TOK_ACT_S)[0]
    d.edge(act_box, act_enc, S_EDGE)
    d.edge(act_enc, act_tok, S_EDGE)

    # ================================================== MoS: mixed start ====
    hist = d.box(40, 710, 180, 70, "Action History h<br>(rtc(h))", S_DASH)
    mix = d.box(280, 705, 190, 85,
                "Mixed Start<br><i>s = β·rtc(h) + (1−β)·z₀ + ηε</i>", S_YEL)
    conf = d.box(280, 820, 190, 55, "Confidence Head β&#770;", S_TEAL)
    d.edge(hist, mix, S_EDGE_TEAL)
    d.edge(z0_tok, mix, S_EDGE_TEAL, pts=[(650, 745)])
    d.edge(conf, mix, S_EDGE_DOT, label="β&#770;")
    mix2act = d.edge(mix, act_box, S_EDGE_TEAL, pts=[(330, 682), (202, 682)])

    # ===================================================== DiT blocks =======
    dit = d.box(730, 330, 360, 260, "", S_CONT)
    ca1 = d.box(748, 350, 55, 200, "Cross-Attention", S_ATTN)
    sa1 = d.box(818, 350, 55, 200, "Self-Attention", S_ATTN)
    d.box(888, 420, 40, 40, "&#8943;",
          "text;html=1;fontSize=28;align=center;verticalAlign=middle;")
    ca2 = d.box(938, 350, 55, 200, "Cross-Attention", S_ATTN)
    sa2 = d.box(1008, 350, 55, 200, "Self-Attention", S_ATTN)
    d.box(878, 592, 60, 24, "× N", S_LBL)
    d.box(860, 616, 100, 24, "DiT Blocks", S_LBL)

    d.edge(vlm_out, ca1, S_EDGE, pts=[(910, 140), (775, 140)])
    d.edge(vlm_out_t, ca2, S_EDGE, pts=[(910, 185), (965, 185)])
    d.edge(st_tok, dit, S_EDGE, pts=[(620, 371), (620, 385), (700, 385)])
    d.edge(act_tok, dit, S_EDGE, pts=[(620, 569), (700, 569), (700, 540)])

    # ===================================================== output side ======
    out_tok = d.token_col(1120, 360, 8, C_TOK_OUT)[0]
    act_dec = d.box(1170, 340, 130, 220, "Action\nDecoder", S_GREEN)
    out_box = d.box(1340, 360, 90, 190,
                    "a<sub>t</sub><br><br>a<sub>t+1</sub><br><br>&#8942;<br><br>a<sub>t+H-1</sub>",
                    S_DASH)
    robot = d.box(1470, 350, 95, 205, "robot", S_PH)
    d.box(1450, 565, 135, 30, "Motor Action", S_TEXT)
    d.edge(dit, out_tok, S_EDGE, pts=[(1105, 460)])
    d.edge(out_tok, act_dec, S_EDGE)
    d.edge(act_dec, out_box, S_EDGE)
    d.edge(out_box, robot, S_EDGE)

    # K iterations loop (offset from the mixed-start edge)
    d.edge(act_dec, act_box, S_EDGE_DOT, label="K iterations",
           pts=[(1235, 697), (243, 697)])

    # ===================================================== legend ==========
    lx, lw = 1290, 26
    d.box(lx, 45, lw, lw, "", f"fillColor={C_GREEN_F};strokeColor={C_GREEN_S};html=1;")
    d.box(lx + 38, 45, 320, lw, "Embodiment-Specific Module", S_LBL_L)
    d.box(lx, 83, lw, lw, "&#10052;",
          "text;html=1;fontSize=18;fontColor=#2D7CFF;align=center;")
    d.box(lx + 38, 83, 320, lw, "Pre-trained and Frozen", S_LBL_L)
    d.box(lx, 121, lw, lw, "", f"fillColor={C_TEAL_F};strokeColor={C_TEAL_S};html=1;")
    d.box(lx + 38, 121, 340, lw, "MoS: Memory Encoder / Confidence Head", S_LBL_L)
    d.box(lx, 159, lw, lw, "", f"fillColor={C_YEL_F};strokeColor={C_YEL_S};html=1;")
    d.box(lx + 38, 159, 340, lw, "MoS: Mixed-Start Node", S_LBL_L)
    return d


# ------------------------------------------------------- PNG compositing ----
def composite(raw_png, out_png, cam_icon, robot_icon):
    """Paste photo icons over the magenta-marked placeholder boxes."""
    im = Image.open(raw_png).convert("RGB")
    a = np.asarray(im)
    mask = (a[..., 0] > 220) & (a[..., 1] < 60) & (a[..., 2] > 220)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("no magenta placeholder found in " + raw_png)

    # split into connected clusters (two placeholders, far apart in x)
    order = np.argsort(xs)
    xs_s, ys_s = xs[order], ys[order]
    splits = np.where(np.diff(xs_s) > 200)[0]
    clusters = np.split(np.arange(len(xs_s)), splits + 1)
    boxes = []
    for cl in clusters:
        xi, yi = xs_s[cl], ys_s[cl]
        boxes.append((xi.min(), yi.min(), xi.max() + 1, yi.max() + 1))
    boxes.sort(key=lambda b: b[2] - b[0], reverse=True)  # camera box is wider

    draw_canvas = im.copy()
    W, H = im.size
    for (x0, y0, x1, y1), icon_path in zip(boxes, [cam_icon, robot_icon]):
        x0, y0 = max(0, x0 - 5), max(0, y0 - 5)
        x1, y1 = min(W, x1 + 5), min(H, y1 + 5)
        icon = Image.open(icon_path).convert("RGB").resize(
            (x1 - x0, y1 - y0), Image.LANCZOS)
        draw_canvas.paste(icon, (x0, y0))
        ImageDraw.Draw(draw_canvas).rectangle(
            [x0, y0, x1 - 1, y1 - 1], outline=(120, 120, 120), width=2)
    draw_canvas.save(out_png)
    print("composited:", out_png, "placeholders:", boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/mos_architecture_mixed_start.drawio")
    ap.add_argument("--cam", default="figures/_cam_icon.png")
    ap.add_argument("--robot", default="figures/_robot_icon.png")
    ap.add_argument("--composite-raw", default=None,
                    help="CLI-exported PNG to composite photos into")
    ap.add_argument("--composite-out", default=None)
    args = ap.parse_args()

    if args.composite_raw:
        composite(args.composite_raw,
                  args.composite_out or
                  args.composite_raw.replace("_raw", ""),
                  args.cam, args.robot)
        return

    d = build()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(d.xml())
    print("written:", args.out)


if __name__ == "__main__":
    main()
