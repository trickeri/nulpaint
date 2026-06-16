# NulPaint in-Krita plugin entry point.
# Krita imports this package and instantiates the Extension.
from .nulpaint import NulPaintExtension

from krita import Krita  # type: ignore  # provided by Krita's interpreter

Krita.instance().addExtension(NulPaintExtension(Krita.instance()))
