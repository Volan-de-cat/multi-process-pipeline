"""Bounded JSON wire protocol: fixed record constructors, never executable pickle.

Datagrams carry scalars, containers, enums and explicitly registered dataclasses.
Decode validates field types before construction; no class/module names supplied
by a sender are imported. Kernel handles remain private to same-trust workers.
"""

import json
import logging
import math
import types
from dataclasses import fields, is_dataclass
from functools import lru_cache
from typing import Any, get_args, get_origin, get_type_hints, is_typeddict

from . import messages
from .validation import validate_record

RECORDS = {
    cls.__name__: cls
    for cls in (
        messages.FrameMetadata,
        messages.FrameDescriptor,
        messages.InferenceResult,
        messages.ProcessingTimings,
        messages.ProcessedResult,
        messages.EndOfStream,
        messages.HealthEvent,
        messages.StageSummary,
    )
}


SCHEMAS = {name: get_type_hints(cls) for name, cls in RECORDS.items()}


@lru_cache(maxsize=32)
def typed_schema(annotation: type) -> dict[str, Any]:
    """Resolve inherited TypedDict fields once; keep the cache bounded."""
    return get_type_hints(annotation)


def matches(value: Any, annotation: Any) -> bool:
    """Check the small supported annotation vocabulary without coercing bad types."""
    if is_typeddict(annotation):
        schema = typed_schema(annotation)
        return (
            isinstance(value, dict)
            and annotation.__required_keys__ <= value.keys() <= schema.keys()
            and all(matches(item, schema[key]) for key, item in value.items())
        )
    origin = get_origin(annotation)
    if origin is types.UnionType:
        return any(matches(value, part) for part in get_args(annotation))
    if origin is dict:
        key_type, item_type = get_args(annotation)
        return isinstance(value, dict) and all(
            matches(key, key_type) and matches(item, item_type) for key, item in value.items()
        )
    if annotation is float:
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False
    if annotation in (int, bool, str, dict, type(None)):
        return type(value) is annotation
    return isinstance(value, annotation)


def pack(value: Any, depth: int = 0) -> Any:
    """Encode only allowed records; cap nesting and strip arbitrary LogRecord extras."""
    if depth > 12:
        raise ValueError("wire nesting limit exceeded")
    if isinstance(value, messages.Stage):
        return {"type": "Stage", "value": value.value}
    if isinstance(value, logging.LogRecord):
        if not matches(value.created, float) or not 0 <= value.created < 253402300799:
            raise ValueError("log timestamp outside supported UTC range")
        return {
            "type": "LogRecord",
            "value": {
                "name": value.name[:128],
                "levelno": value.levelno,
                "message": value.getMessage()[:8192],
                "created": value.created,
                "processName": str(value.processName)[:128],
            },
        }
    if is_dataclass(value) and type(value) in RECORDS.values():
        schema = SCHEMAS[type(value).__name__]
        if not all(matches(getattr(value, key), annotation) for key, annotation in schema.items()):
            raise ValueError("wire record field types differ from schema")
        validate_record(value)
        return {
            "type": type(value).__name__,
            "value": {f.name: pack(getattr(value, f.name), depth + 1) for f in fields(value)},
        }
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("wire dictionary keys must be strings")
        return {"type": "dict", "value": {k: pack(v, depth + 1) for k, v in value.items()}}
    if type(value) in (list, tuple):
        return {"type": "list", "value": [pack(v, depth + 1) for v in value]}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("unsupported wire value")


def unpack(value: Any, depth: int = 0) -> Any:
    """Validate envelopes and record schemas, rejecting unknown tags and extra fields."""
    if depth > 12:
        raise ValueError("wire nesting limit exceeded")
    if not isinstance(value, dict):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("nonfinite wire value")
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise ValueError("invalid wire scalar")
    if set(value) != {"type", "value"} or type(value["type"]) is not str:
        raise ValueError("invalid wire envelope")
    tag, body = value["type"], value["value"]
    if tag == "Stage":
        return messages.Stage(body)
    if tag == "list" and isinstance(body, list):
        return [unpack(v, depth + 1) for v in body]
    if tag == "dict" and isinstance(body, dict):
        return {k: unpack(v, depth + 1) for k, v in body.items()}
    if tag == "LogRecord" and isinstance(body, dict):
        return unpack_log(body)
    return unpack_record(tag, body, depth)


def encode(value: Any) -> bytes:
    """Serialize a validated bounded record to UTF-8; reject NaN and infinity."""
    return json.dumps(pack(value), allow_nan=False, separators=(",", ":")).encode("utf-8")


def decode(payload: bytes) -> Any:
    """Decode UTF-8 JSON with a bounded structure; malformed input fails closed."""
    if len(payload) > 60_000:
        raise ValueError("wire payload exceeds byte limit")
    try:
        return unpack(
            json.loads(
                payload,
                parse_constant=reject_constant,
                parse_float=finite_float,
                object_pairs_hook=unique_fields,
            )
        )
    except (RecursionError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid wire payload") from exc


def reject_constant(value: str) -> Any:
    """Reject nonstandard JSON constants rather than materializing NaN/Infinity."""
    raise ValueError(f"invalid numeric constant: {value}")


def finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite JSON number")
    return result


def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def unpack_log(body: dict[str, Any]) -> logging.LogRecord:
    """Reconstruct only the whitelisted scalar fields of a service log."""
    expected = {
        "name": str,
        "levelno": int,
        "message": str,
        "created": float,
        "processName": str,
    }
    if set(body) != set(expected) or not all(matches(body[k], t) for k, t in expected.items()):
        raise ValueError("invalid log record")
    if not 0 <= body["created"] < 253402300799:
        raise ValueError("log timestamp outside supported UTC range")
    record = logging.LogRecord(body["name"], body["levelno"], "", 0, body["message"], (), None)
    record.created, record.processName = body["created"], body["processName"]
    return record


def unpack_record(tag: str, body: Any, depth: int) -> Any:
    """Validate one registered dataclass against its cached schema."""
    if tag not in RECORDS or not isinstance(body, dict):
        raise ValueError("unknown wire record")
    cls = RECORDS[tag]
    expected = SCHEMAS[tag]
    if set(body) != set(expected):
        raise ValueError("wire record fields differ from schema")
    decoded = {k: unpack(v, depth + 1) for k, v in body.items()}
    if not all(matches(decoded[k], t) for k, t in expected.items()):
        raise ValueError("wire record field types differ from schema")
    record = cls(**decoded)
    validate_record(record)
    return record
