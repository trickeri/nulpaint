# NulPaint in-Krita plugin entry point.
# Krita imports this package and instantiates the Extension + registers dockers.
from .nulpaint import NulPaintExtension

from krita import Krita, DockWidgetFactory, DockWidgetFactoryBase  # type: ignore

Krita.instance().addExtension(NulPaintExtension(Krita.instance()))

from .ai_docker import NulPaintAIDocker  # noqa: E402

Krita.instance().addDockWidgetFactory(DockWidgetFactory(
    "nulpaint_ai", DockWidgetFactoryBase.DockPosition.DockRight, NulPaintAIDocker))
