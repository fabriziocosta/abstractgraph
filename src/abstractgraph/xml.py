"""
XML round-trip for AbstractGraph operator pipelines.

Features
- Serialises operator pipelines: `compose`, `forward_compose`, `add`, `compose_product`.
- Handles conditional/iterative operators: `if_then_else`, `if_then_elif_else`, `for_loop`, `while_loop`.
- Works with toolz.curry-annotated operators: captures bound kwargs and callable args.
- Safe param values: only Python literals are embedded directly; callable kwargs are saved as references.
- Pluggable registries to resolve operators and combiners.

Usage
    from coco_grape.module.abstract_graph import operator as qg_ops
    from coco_grape.module.abstract_graph.abstract_graph_xml import register_from_module,
        operator_to_xml_string, operator_from_xml_string

    register_from_module(qg_ops)
    xml = operator_to_xml_string(pipeline)
    pipeline2 = operator_from_xml_string(xml)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional
import ast
import xml.etree.ElementTree as ET

# --------------------------------------------------------------------------------------
# Registries
# --------------------------------------------------------------------------------------

OPERATOR_REGISTRY: Dict[str, Callable[..., Any]] = {}
COMBINER_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_operator(name: Optional[str] = None):
    """
    Decorator to register an operator constructor by name.

    Args:
        name: Optional explicit registry key.

    Returns:
        Callable: Decorator that registers a callable in OPERATOR_REGISTRY.
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        key = name or getattr(func, "__name__", None)
        if not key:
            raise ValueError("Cannot register operator without a name")
        OPERATOR_REGISTRY[key] = func
        return func

    return _decorator


def register_combiner(name: str, func: Callable[..., Any]) -> None:
    """
    Register a combiner function for compose_product round-trips.

    Args:
        name: Registry key for the combiner.
        func: Combiner callable.

    Returns:
        None.
    """
    if not name:
        raise ValueError("Combiner name must be non-empty")
    COMBINER_REGISTRY[name] = func


def register_from_module(module: Any, names: Optional[Iterable[str]] = None) -> None:
    """
    Bulk-register callables from a module by attribute name.

    Args:
        module: Module object to inspect.
        names: Optional iterable of attribute names to register.

    Returns:
        None.
    """
    cand_names = names or [n for n in dir(module) if not n.startswith("_")]
    for n in cand_names:
        obj = getattr(module, n, None)
        if callable(obj):
            OPERATOR_REGISTRY[n] = obj


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _is_curry_obj(obj: Any) -> bool:
    """
    Check if an object looks like a toolz.curry wrapper.

    Args:
        obj: Object to inspect.

    Returns:
        bool: True if the object has curry-like attributes.
    """
    return hasattr(obj, "func") and hasattr(obj, "args") and hasattr(obj, "keywords")


def _op_name(op: Any) -> str:
    """Resolve an operator's name for XML `type` attribute.

    Prefer `__name__` so that `compose_product` round-trips, since some composed
    ops in this codebase set `operator_type = "product"` but `__name__ = "compose_product"`.
    Args:
        op: Operator or callable to name.

    Returns:
        str: Name used for serialization.
    """
    if hasattr(op, "__name__") and isinstance(op.__name__, str) and op.__name__:
        return op.__name__
    if hasattr(op, "operator_type") and isinstance(op.operator_type, str) and op.operator_type:
        # Map legacy/product tag to constructor name for round-trip
        return "compose_product" if op.operator_type == "product" else op.operator_type
    if _is_curry_obj(op) and hasattr(op, "func") and hasattr(op.func, "__name__"):
        return op.func.__name__
    return type(op).__name__


def _op_bound_kwargs(op: Any) -> Dict[str, Any]:
    """
    Extract bound keyword parameters from an operator.

    Args:
        op: Operator or curry-wrapped callable.

    Returns:
        Dict[str, Any]: Bound keyword arguments.
    """
    if hasattr(op, "params") and isinstance(getattr(op, "params"), dict):
        return dict(getattr(op, "params"))
    if _is_curry_obj(op) and hasattr(op, "keywords"):
        return dict(op.keywords or {})
    return {}


def _op_children(op: Any) -> List[Any]:
    """
    Extract child operators for composite operators.

    Args:
        op: Operator or composite callable.

    Returns:
        List[Any]: Child operators.
    """
    for attr in ("children", "chain", "decomposition_functions"):
        if hasattr(op, attr):
            seq = getattr(op, attr)
            return list(seq) if seq else []
    if _is_curry_obj(op) and getattr(op, "args", None):
        return [a for a in op.args if callable(a)]
    return []


def _maybe_combiner_name(op: Any) -> Optional[str]:
    """
    Resolve a combiner name for compose_product operators.

    Args:
        op: Operator possibly carrying a combiner.

    Returns:
        Optional[str]: Combiner registry name or function name.
    """
    name = getattr(op, "combiner_name", None)
    if isinstance(name, str) and name:
        return name
    comb = getattr(op, "combiner", None)
    if comb is None:
        return None
    for k, v in COMBINER_REGISTRY.items():
        if v is comb:
            return k
    return getattr(comb, "__name__", None)


def _to_attr_value(value: Any) -> str:
    """
    Serialize a Python literal to an attribute string via repr.

    Args:
        value: Value to serialize.

    Returns:
        str: Serialized representation.
    """
    return repr(value)


def _from_attr_value(text: str) -> Any:
    """
    Parse a serialized attribute string using literal_eval.

    Args:
        text: Serialized attribute value.

    Returns:
        Any: Parsed value.
    """
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _encode_param_value(value: Any) -> str:
    """Encode parameter values.
    - Callables are stored as "ref:<name>" and looked up in the operator registry.
    - Literals use repr so literal_eval can restore them.
    Args:
        value: Parameter value to encode.

    Returns:
        str: Encoded parameter string.
    """
    if callable(value):
        return f"ref:{_op_name(value)}"
    return _to_attr_value(value)


def _decode_param_value(value: str) -> Any:
    """
    Decode a parameter value string to a Python value or callable.

    Args:
        value: Encoded parameter value.

    Returns:
        Any: Decoded parameter value.
    """
    if isinstance(value, str) and value.startswith("ref:"):
        ref_name = value.split(":", 1)[1]
        if ref_name not in OPERATOR_REGISTRY:
            raise KeyError(f"Unknown referenced callable '{ref_name}'. Register it first.")
        return OPERATOR_REGISTRY[ref_name]
    return _from_attr_value(value)


def _apply_metadata(
    op: Any,
    annotation: Optional[str],
    reason_synthesis: Optional[str],
    name: Optional[str],
) -> Any:
    """
    Attach optional metadata attributes to an operator.

    Args:
        op: Operator to annotate.
        annotation: Optional annotation string.
        reason_synthesis: Optional reason synthesis string.
        name: Optional operator name override.

    Returns:
        Any: Operator with metadata attached when possible.
    """
    if name:
        try:
            op.name = name  # type: ignore[attr-defined]
        except Exception:
            pass
    if annotation:
        try:
            op.annotation = annotation  # type: ignore[attr-defined]
        except Exception:
            pass
    if reason_synthesis:
        try:
            op.reason_synthesis = reason_synthesis  # type: ignore[attr-defined]
        except Exception:
            pass
    return op


# --------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------

def operator_to_xml_element(op: Any) -> ET.Element:
    """
    Serialize an operator into an XML element.

    Args:
        op: Operator to serialize.

    Returns:
        ET.Element: XML element representing the operator.
    """
    elem = ET.Element("Operator")
    elem.set("type", _op_name(op))

    name = getattr(op, "name", None)
    if isinstance(name, str) and name:
        elem.set("name", name)

    annotation = getattr(op, "annotation", None)
    if isinstance(annotation, str) and annotation:
        ann_elem = ET.SubElement(elem, "Annotation")
        ann_elem.text = annotation
    reason_synthesis = getattr(op, "reason_synthesis", None)
    if isinstance(reason_synthesis, str) and reason_synthesis:
        reason_elem = ET.SubElement(elem, "ReasonSynthesis")
        reason_elem.text = reason_synthesis

    # Parameters (kwargs). Encode callables as refs.
    for k, v in sorted(_op_bound_kwargs(op).items()):
        elem.set(k, _encode_param_value(v))

    # combiner for compose_product
    # Prefer legacy attribute when the combiner is a simple named function registered via COMBINER_REGISTRY.
    # Otherwise (e.g., curried/parameterised operators like binary_combination(distance=...)),
    # serialise as a nested <Combiner><Operator .../></Combiner> element for full round-trip of params.
    combiner_obj = getattr(op, "combiner", None)
    if combiner_obj is not None:
        bound_kwargs = _op_bound_kwargs(combiner_obj)
        is_curried = _is_curry_obj(combiner_obj)
        combiner_name = _maybe_combiner_name(op)

        use_legacy_attr = (
            isinstance(combiner_name, str)
            and combiner_name in COMBINER_REGISTRY
            and not bound_kwargs
            and not is_curried
        )

        if use_legacy_attr:
            elem.set("combiner", combiner_name)
        else:
            combiner_elem = ET.SubElement(elem, "Combiner")
            combiner_elem.append(operator_to_xml_element(combiner_obj))

    # Children (composition operands, or positional callables for loops, etc.)
    for child in _op_children(op):
        child_elem = ET.SubElement(elem, "Child")
        child_elem.append(operator_to_xml_element(child))

    return elem


def operator_to_xml_string(op: Any, pretty: bool = True) -> str:
    """
    Serialize an operator pipeline to an XML string.

    Args:
        op: Operator to serialize.
        pretty: Whether to pretty-print XML.

    Returns:
        str: XML string representation.
    """
    elem = operator_to_xml_element(op)
    xml = ET.tostring(elem, encoding="unicode")
    if not pretty:
        return xml
    try:
        import xml.dom.minidom as minidom
        return minidom.parseString(xml).toprettyxml(indent="  ")
    except Exception:
        return xml


# --------------------------------------------------------------------------------------
# Deserialisation
# --------------------------------------------------------------------------------------

def _resolve_operator_constructor(name: str) -> Callable[..., Any]:
    """
    Look up an operator constructor by name.

    Args:
        name: Registry key for the operator.

    Returns:
        Callable[..., Any]: Operator constructor.
    """
    if name not in OPERATOR_REGISTRY:
        raise KeyError(f"Unknown operator type '{name}'. Register it first.")
    return OPERATOR_REGISTRY[name]


def _resolve_combiner(name: str) -> Callable[..., Any]:
    """
    Look up a combiner by name.

    Args:
        name: Registry key for the combiner.

    Returns:
        Callable[..., Any]: Combiner callable.
    """
    if name not in COMBINER_REGISTRY:
        raise KeyError(f"Unknown combiner '{name}'. Register it first.")
    return COMBINER_REGISTRY[name]


def _build_wrapped(
    name: str,
    builder: Callable[..., Any],
    children: List[Any],
    params: Dict[str, Any],
) -> Callable[[Any], Any]:
    """
    Wrap a constructor into a callable op expecting an AbstractGraph first argument.

    Args:
        name: Operator name for metadata.
        builder: Operator constructor callable.
        children: Child operators to bind.
        params: Keyword parameters to bind.

    Returns:
        Callable[[Any], Any]: Wrapped operator callable.
    """

    def _op(abstract_graph):
        return builder(abstract_graph, *children, **params)

    _op.__name__ = name
    _op.operator_type = name  # type: ignore[attr-defined]
    _op.children = list(children)  # type: ignore[attr-defined]
    _op.params = dict(params)  # type: ignore[attr-defined]
    return _op


def operator_from_xml_element(elem: ET.Element) -> Any:
    """
    Deserialize an operator from an XML element.

    Args:
        elem: XML element describing the operator.

    Returns:
        Any: Deserialized operator.
    """
    op_type = elem.attrib.get("type")
    if not op_type:
        raise ValueError("Operator element missing 'type' attribute")

    # Parameters
    params: Dict[str, Any] = {}
    for k, v in elem.attrib.items():
        if k in ("type", "combiner", "name"):
            continue
        params[k] = _decode_param_value(v)

    # Children
    child_ops: List[Any] = []
    for child in elem.findall("Child"):
        inner = child.find("Operator")
        if inner is None:
            raise ValueError("Child element missing nested Operator")
        child_ops.append(operator_from_xml_element(inner))

    annotation_elem = elem.find("Annotation")
    annotation = annotation_elem.text if annotation_elem is not None else None
    reason_elem = elem.find("ReasonSynthesis")
    reason_synthesis = reason_elem.text if reason_elem is not None else None
    op_name = elem.attrib.get("name")

    # Compose variants and add
    if op_type in ("compose", "forward_compose", "add"):
        builder = _resolve_operator_constructor(op_type)
        op = builder(*child_ops, **params)
        return _apply_metadata(op, annotation, reason_synthesis, op_name)

    # Product composition with combiner
    if op_type == "compose_product" or op_type == "product":  # accept legacy tag
        builder = _resolve_operator_constructor("compose_product")

        # Prefer legacy attribute path when present
        comb_name = elem.attrib.get("combiner")
        comb = None
        if comb_name:
            comb = _resolve_combiner(comb_name)
        else:
            # Look for nested <Combiner><Operator/></Combiner>
            combiner_container = elem.find("Combiner")
            if combiner_container is None:
                raise ValueError("compose_product element missing combiner (no 'combiner' attribute or <Combiner> child)")
            inner = combiner_container.find("Operator")
            if inner is None:
                raise ValueError("<Combiner> element missing nested <Operator>")
            comb = operator_from_xml_element(inner)

        op = builder(comb, *child_ops, **params)
        if comb_name:
            try:
                op.combiner_name = comb_name  # type: ignore[attr-defined]
            except Exception:
                pass
        return _apply_metadata(op, annotation, reason_synthesis, op_name)

    # Conditional and loops — return a wrapped callable that receives ag first
    if op_type in ("if_then_else", "if_then_elif_else", "for_loop", "while_loop"):
        builder = _resolve_operator_constructor(op_type)
        op = _build_wrapped(op_type, builder, child_ops, params)
        return _apply_metadata(op, annotation, reason_synthesis, op_name)

    # Leaf/general curried operators
    builder = _resolve_operator_constructor(op_type)
    try:
        op = builder(**params) if params else builder()
    except TypeError:
        # Fallback if builder expects the ag first (non-curried)
        op = _build_wrapped(op_type, builder, [], params)
    try:
        op.operator_type = op_type  # type: ignore[attr-defined]
        if params:
            op.params = dict(params)  # type: ignore[attr-defined]
    except Exception:
        pass
    return _apply_metadata(op, annotation, reason_synthesis, op_name)


def operator_from_xml_string(xml: str) -> Any:
    """
    Deserialize an operator pipeline from an XML string.

    Args:
        xml: XML string to parse.

    Returns:
        Any: Deserialized operator.
    """
    root = ET.fromstring(xml)
    if root.tag != "Operator":
        raise ValueError("Root element must be <Operator>")
    return operator_from_xml_element(root)


def operator_to_xml_file(op: Any, path: str, pretty: bool = True) -> None:
    """
    Serialize an operator pipeline to a file.

    Args:
        op: Operator to serialize.
        path: Output file path.
        pretty: Whether to pretty-print XML.

    Returns:
        None.
    """
    xml = operator_to_xml_string(op, pretty=pretty)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)


def operator_from_xml_file(path: str) -> Any:
    """
    Deserialize an operator pipeline from a file.

    Args:
        path: XML file path.

    Returns:
        Any: Deserialized operator.
    """
    with open(path, "r", encoding="utf-8") as f:
        xml = f.read()
    return operator_from_xml_string(xml)


def annotate_operator(op: Any, annotation: str) -> Any:
    """
    Attach an annotation string to an operator.

    Args:
        op: Operator to annotate.
        annotation: Annotation text.

    Returns:
        Any: Annotated operator.
    """
    op.annotation = annotation  # type: ignore[attr-defined]
    return op
