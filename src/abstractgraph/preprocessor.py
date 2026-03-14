"""Compatibility shim for the extracted preprocessor implementation.

`abstractgraph.preprocessor` now re-exports the attention-driven graph
induction classes from `abstractgraph-graphicalizer` during the staged
compatibility window.
"""

from abstractgraph_graphicalizer.attention import AbstractGraphPreprocessor, ImageNodeClusterer

__all__ = ["AbstractGraphPreprocessor", "ImageNodeClusterer"]
