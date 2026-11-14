"""The tables of the contract reference (docs/reference/ in Chinese, docs/en/reference/ in English),
written from the golden copies in tests/contracts/, so the reference says what the tests hold.

Each table sits between `<!-- generated: NAME -->` and `<!-- end generated -->`; everything else in
the files is written by hand. tests/test_reference.py fails when a table is out of date.

    uv run sotaigen-dev reference [--check]
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "contracts"
FOLDERS = {"zh": ROOT / "docs" / "reference", "en": ROOT / "docs" / "en" / "reference"}
BLOCK = re.compile(r"(<!-- generated: ([\w-]+) -->\n)(.*?)(<!-- end generated -->)", re.DOTALL)

# What a parameter is measured on (sotaigen.params.spec.BASES), in each language.
BASES = {
    "en": {
        "skeleton": "between landmarks the bones are placed from",
        "surface": "the finished surface; solved for",
        "calibrated": "the surface's breadth at its landmark, within 2%",
        "chord": "the surface's offset from a chord between two stations",
        "without": "how far the surface moves against the same body without it",
        "under": "the most the tissue stands off the surface under it",
        "shape": "a dimension of the shape it adds",
        "share": "a ratio of two dimensions of one part",
        "control": "an internal control; what a value makes may change with the shape's version",
    },
    "zh": {
        "skeleton": "骨骼据以放置的地标之间",
        "surface": "成品曲面，由求解器反求",
        "calibrated": "曲面在地标处的宽度，误差 2% 以内",
        "chord": "曲面离两处之间弦线的偏移",
        "without": "曲面比不加这一项的同一身体移动多少",
        "under": "组织离其下曲面最远多少",
        "shape": "这一项所加形状的尺寸",
        "share": "同一部位两个尺寸之比",
        "control": "内部控制量，同一个值做出什么随形体版本变",
    },
}
# Parameters without a slider of their own, so without a label in web/messages.
LABELS = {
    "waistRise": {"en": "Waist height above the crotch", "zh": "腰地标在裆上的高度"},
    "waistDepth": {"en": "Waist depth, front to back at the waist landmark", "zh": "腰厚，腰地标高度上的前后距离"},
    "definition": {"en": "Definition of the muscle masses", "zh": "肌肉体块的分明程度"},
    "softness": {"en": "Softness of the blends between masses", "zh": "体块之间融合的柔和程度"},
}
# The part a slider's label leaves to its group on the page.
PARTS = (
    (("glute", "outerHip"), {"en": "Glutes", "zh": "臀部"}),
    (("breast",), {"en": "Breasts", "zh": "胸部"}),
    (("groin",), {"en": "Groin", "zh": "腹股沟"}),
    (("outerThigh", "thighFront"), {"en": "Thigh", "zh": "大腿"}),
    (("abdomen", "lumbar", "waistContour"), {"en": "Belly and back", "zh": "腰腹"}),
)
WORDS = {
    "en": {
        "parameter": "Parameter", "unit": "Unit", "required": "Required", "basis": "Measured on",
        "what": "What it is", "yes": "yes", "no": "no", "bone": "Bone", "parent": "Parent",
        "command": "Command", "arguments": "Arguments", "code": "Code", "values": "Values",
        "meaning": "Meaning", "exit": "Exit code", "name": "Name", "mm": "mm", "ratio": "ratio",
        "exitMeaning": {
            "done": "done", "refused": "the body can't be made or written: a rule, a broken mesh",
            "usage": "a wrong command line or file",
            "missing": "Blender missing or older than the floor, or sotaigen-blender missing",
        },
        "what_for": "What", "value": "Value",
    },
    "zh": {
        "parameter": "参数", "unit": "单位", "required": "必填", "basis": "参照面", "what": "是什么",
        "yes": "是", "no": "否", "bone": "骨", "parent": "父骨", "command": "子命令", "arguments": "参数",
        "code": "代码", "values": "`values` 的键", "meaning": "含义", "exit": "退出码", "name": "名字",
        "mm": "mm", "ratio": "比值",
        "exitMeaning": {
            "done": "完成", "refused": "身体做不成或写不出：规则不过、网格损坏",
            "usage": "命令行或文件不对", "missing": "缺 Blender、Blender 低于最低版本，或缺 sotaigen-blender",
        },
        "what_for": "东西", "value": "值",
    },
}
BLEND_ROWS = {
    "objects.rig": {"en": "The armature object", "zh": "骨架物体"},
    "objects.body": {"en": "The skin mesh object", "zh": "皮肤网格物体"},
    "objects.mechanism": {"en": "The empty holding the correctives' intermediate properties", "zh": "放修正形状中间属性的空物体"},
    "controlSuffix": {"en": "A control bone is its deform bone's name plus this", "zh": "控制骨是变形骨名加这个后缀"},
    "widgetPrefix": {"en": "The control shapes' prefix", "zh": "控件形状的前缀"},
    "vertexGroups.headRegion": {"en": "Vertex group: the head's skin", "zh": "顶点组：头的皮肤"},
    "vertexGroups.neckRing": {"en": "Vertex group: the neck ring", "zh": "顶点组：颈环"},
    "properties.jointLimits": {"en": "Custom property switching the joint limits", "zh": "开关关节极限的自定义属性"},
    "properties.info": {"en": "Custom property on the armature: what the GLB's extras.sotaigen says", "zh": "骨架上的自定义属性，内容同 GLB 的 `extras.sotaigen`"},
    "collections.controls": {"en": "Bone collection of the controls", "zh": "控制骨的骨骼集合"},
    "collections.deform": {"en": "Bone collection of the deform bones", "zh": "变形骨的骨骼集合"},
    "collections.parts": {"en": "Collection of the placeholder parts", "zh": "占位部件的集合"},
    "collections.mechanism": {"en": "Collection of the mechanism", "zh": "机制的集合"},
    "collections.widgets": {"en": "Collection of the widgets", "zh": "控件形状的集合"},
}


def golden(name):
    return json.loads((GOLDEN / f"{name}.json").read_text())


def messages(language):
    return json.loads((ROOT / "web" / "messages" / f"{language}.json").read_text())


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def parameters(language, sex):
    w, said = WORDS[language], messages(language)
    rows = []
    for key, row in golden("parameters")["sexes"][sex].items():
        label = LABELS.get(key, {}).get(language) or said[f"slider_{key}"]
        part = next((names[language] for starts, names in PARTS if key.startswith(starts)), None)
        if part:
            label = f"{part}: {label}" if language == "en" else f"{part}：{label}"
        rows.append((f"`{key}`", w[row["unit"]], w["no"] if row["optional"] else w["yes"],
                     BASES[language][row["basis"]], label))
    return table((w["parameter"], w["unit"], w["required"], w["basis"], w["what"]), rows)


def bones(language):
    w = WORDS[language]
    rows = [(i, f"`{b['name']}`", f"`{b['parent']}`" if b["parent"] else "—") for i, b in enumerate(golden("bones"))]
    return table(("#", w["bone"], w["parent"]), rows)


def bodies(language):
    return "".join(f"- `{name}`\n" for name in golden("bodies")["typical"])


def cli(language):
    w, found = WORDS[language], golden("cli")

    def argument(a):
        return f"`{a}`" if isinstance(a, str) else " / ".join(f"`{flag}`" for flag in a)

    commands = table((w["command"], w["arguments"]),
                     [(f"`sotaigen {name}`", ", ".join(map(argument, args))) for name, args in found["commands"].items()])
    codes = table((w["exit"], w["meaning"]),
                  [(code, w["exitMeaning"][name]) for name, code in found["exitCodes"].items()])
    return commands + "\n" + codes


def diagnostics(language):
    w, found = WORDS[language], golden("diagnostics")
    return table((w["code"], w["values"]),
                 [(f"`{code}`", ", ".join(f"`{v}`" for v in values) or "—") for code, values in found["codes"].items()])


def blend_names(language):
    w, names = WORDS[language], golden("blend")["names"]
    rows = []
    for key, value in names.items():
        if isinstance(value, dict):
            rows += [(BLEND_ROWS[f"{key}.{sub}"][language], f"`{v}`") for sub, v in value.items()]
        elif isinstance(value, str):
            rows.append((BLEND_ROWS[key][language], f"`{value}`"))
    constraints = {"en": "Constraints, by what they do", "zh": "约束，按作用命名"}[language]
    collections = {"en": "Bone collections under the controls", "zh": "控制骨下的骨骼集合"}[language]
    rows.append((constraints, ", ".join(f"`{c}`" for c in names["constraints"])))
    rows.append((collections, ", ".join(f"`{c}`" for c in names["boneCollections"])))
    return table((w["what_for"], w["name"]), rows)


def glb_groups(language):
    found = golden("glb")
    lines = [
        ("correctives.groups[].name", found["correctiveGroups"]),
        ("quality.checked", found["qualityChecks"]),
        ("expressions", found["expressions"]),
    ]
    w = WORDS[language]
    return table((w["what_for"], w["value"]), [(f"`{k}`", ", ".join(f"`{v}`" for v in vs)) for k, vs in lines])


def pose_assets(language):
    return ", ".join(f"`{name}`" for name in golden("blend")["file"]["poseAssets"]) + "\n"


def blend_info(language):
    return ", ".join(f"`{name}`" for name in golden("blend")["file"]["info"]) + "\n"


def topology(language):
    found = golden("topology")
    start, count = found["headRegion"]
    words = {
        "en": ("Vertices", "Faces", "SHA-256 of the face list", "Head region", "Neck ring", "Crown vertex",
               f"{count} from {start}", f"{len(found['neckRing'])} vertices"),
        "zh": ("顶点", "面", "面列表的 SHA-256", "头的区块", "颈环", "头顶的顶点",
               f"从 {start} 起 {count} 个", f"{len(found['neckRing'])} 个顶点"),
    }[language]
    w = WORDS[language]
    rows = [(words[0], found["vertices"]), (words[1], found["faces"]), (words[2], f"`{found['facesSha256']}`"),
            (words[3], words[6]), (words[4], words[7]), (words[5], found["crownVertex"])]
    return table((w["what_for"], w["value"]), rows)


TABLES = {
    "parameters-female": lambda language: parameters(language, "female"),
    "parameters-male": lambda language: parameters(language, "male"),
    "bones": bones,
    "bodies": bodies,
    "cli": cli,
    "diagnostics": diagnostics,
    "blend-names": blend_names,
    "blend-info": blend_info,
    "pose-assets": pose_assets,
    "glb-values": glb_groups,
    "topology": topology,
}


def rewritten(path, language):
    text = path.read_text()

    def fill(match):
        return match[1] + TABLES[match[2]](language) + match[4]

    return BLOCK.sub(fill, text)


def files():
    for language, folder in FOLDERS.items():
        for path in sorted(folder.glob("*.md")):
            yield language, path


def stale():
    """The reference files whose generated tables aren't what the golden copies say."""
    return [path for language, path in files() if rewritten(path, language) != path.read_text()]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--check" in argv:
        found = stale()
        print("\n".join(str(p.relative_to(ROOT)) for p in found) or "reference ok")
        return 1 if found else 0
    for language, path in files():
        path.write_text(rewritten(path, language))
    print("reference written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
