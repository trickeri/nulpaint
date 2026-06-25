# Move every "<X> (cut)" layer next to its source "<X>", inside the source's group.
# Run from Krita: Tools > Scripts > Scripter, paste this, press Run (the green ▶).
# Safe + idempotent: it rebuilds each cut inside the right group (above its source)
# from the cut's own pixels, then removes the misplaced original. Sources untouched.
from krita import Krita

doc = Krita.instance().activeDocument()
root = doc.rootNode()
W, H = doc.width(), doc.height()
SUF = " (cut)"

def find(node, name):
    for ch in node.childNodes():
        if ch.name() == name:
            return ch
        r = find(ch, name)
        if r:
            return r
    return None

def collect_cuts(node, acc):
    for ch in node.childNodes():
        if ch.type() == "paintlayer" and ch.name().endswith(SUF):
            acc.append(ch)
        collect_cuts(ch, acc)

cuts = []
collect_cuts(root, cuts)

moved, skipped = 0, []
for cut in cuts:
    src = find(root, cut.name()[:-len(SUF)])
    if src is None or src.parentNode() is None:
        skipped.append(cut.name())
        continue
    group = src.parentNode()
    new = doc.createNode(cut.name(), "paintlayer")
    group.addChildNode(new, src)              # new sits directly ABOVE its source
    new.setPixelData(cut.pixelData(0, 0, W, H), 0, 0, W, H)
    new.setVisible(cut.visible())
    cut.parentNode().removeChildNode(cut)     # drop the misplaced original
    moved += 1

doc.refreshProjection()
print("regroup_cuts: moved %d cut layers into their groups; skipped: %s"
      % (moved, skipped or "none"))
