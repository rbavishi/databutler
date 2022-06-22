from typing import Dict, Any, Union, List

import attrs
from mypy import types
from mypy.types import Type as MypyType


def process_typ_str(typ_str: str) -> str:
    mapping = {
        "builtins.list": "List",
        "builtins.dict": "Dict",
        "builtins.set": "Set",
        "builtins.tuple": "Tuple",
        "builtins.str": "str",
        "builtins.int": "int",
        "builtins.float": "float",
        "builtins.bool": "bool",
    }
    for k, v in mapping.items():
        typ_str = typ_str.replace(k, v)

    return typ_str


@attrs.define
class SerializedMypyType:
    type_json: Union[str, Dict[str, Any]]

    @classmethod
    def from_mypy_type(cls, mypy_type: MypyType) -> "SerializedMypyType":
        return SerializedMypyType(mypy_type.serialize())

    def as_mypy(self) -> MypyType:
        if isinstance(self.type_json, str):
            return types.Instance.deserialize(self.type_json)
        else:
            return getattr(types, self.type_json[".class"]).deserialize(self.type_json)

    def is_literal(self) -> bool:
        if isinstance(self.type_json, str):
            return False

        t_json = self.type_json
        if t_json[".class"] == "LiteralType":
            return True

        if (
            t_json[".class"] == "Instance"
            and t_json.get("last_known_value", None) is not None
        ):
            return True

        return False

    def get_literal_value(self) -> Union[str, int, bool]:
        assert isinstance(self.type_json, dict)
        if self.type_json[".class"] == "LiteralType":
            return self.type_json["value"]

        if self.type_json[".class"] == "Instance":
            return self.type_json["last_known_value"]["value"]

        raise ValueError(f"Cannot extract literal value from {self.type_json}")

    def is_any_type(self) -> bool:
        return (
            isinstance(self.type_json, dict) and self.type_json[".class"] == "AnyType"
        )

    def is_module_type(self) -> bool:
        if isinstance(self.type_json, str) and self.type_json == "types.ModuleType":
            return True

        if isinstance(self.type_json, dict) and self.type_json[".class"] == "AnyType":
            if self.type_json.get("type_of_any", None) == 3:
                return True
            if self.type_json.get("missing_import_name", None) is not None:
                return True

        return False

    def is_overloaded_callable_type(self) -> bool:
        return (
            isinstance(self.type_json, dict)
            and self.type_json[".class"] == "Overloaded"
        )

    def is_nonoverloaded_callable_type(self) -> bool:
        return (
            isinstance(self.type_json, dict)
            and self.type_json[".class"] == "CallableType"
        )

    def is_callable_type(self) -> bool:
        return (
            self.is_nonoverloaded_callable_type() or self.is_overloaded_callable_type()
        )

    def unpack_overloaded_callable_type(self) -> List["SerializedMypyType"]:
        if self.is_overloaded_callable_type():
            return [SerializedMypyType(i) for i in self.type_json["items"]]
        else:
            return [self]

    def has_starred_args(self) -> bool:
        if self.is_nonoverloaded_callable_type():
            typ = self.as_mypy()
            assert isinstance(typ, types.CallableType)
            return (typ.var_arg() is not None) or (typ.kw_arg() is not None)
        else:
            return any(
                i.has_starred_args() for i in self.unpack_overloaded_callable_type()
            )

    def get_callable_arg_order(self) -> Dict[str, int]:
        if self.is_nonoverloaded_callable_type():
            names = [i for i in self.type_json["arg_names"] if i is not None]
            return {name: idx for idx, name in enumerate(names)}
        else:
            return self.unpack_overloaded_callable_type()[0].get_callable_arg_order()

    def is_union_type(self) -> bool:
        return (
            isinstance(self.type_json, dict) and self.type_json[".class"] == "UnionType"
        )

    def unpack_union_type(self) -> List["SerializedMypyType"]:
        assert self.is_union_type()
        return [SerializedMypyType(type_json=i) for i in self.type_json["items"]]

    def equals(self, other: Union[str, Dict, MypyType, "SerializedMypyType"]) -> bool:
        if self.is_union_type():
            return any(i.equals(other) for i in self.unpack_union_type())

        if isinstance(other, str):
            if isinstance(self.type_json, str):
                return self.type_json == other
            elif isinstance(self.type_json, dict):
                if self.type_json[".class"] == "Instance":
                    return self.type_json.get("type_ref", None) == other

        elif isinstance(self.type_json, dict):
            return self.type_json == other

        elif isinstance(other, SerializedMypyType):
            return self.type_json == other.type_json

        elif isinstance(other, MypyType):
            return self.type_json == other.serialize()

        return False

    def is_bool_type(self) -> bool:
        return self.equals("builtins.bool")

    def is_int_type(self) -> bool:
        return self.equals("builtins.int")

    def is_str_type(self) -> bool:
        return self.equals("builtins.str")

    def is_float_type(self) -> bool:
        return self.equals("builtins.float")

    def _to_string_type_json(self, type_json: Union[str, Dict]) -> str:
        if isinstance(type_json, str):
            return process_typ_str(type_json)
        elif type_json[".class"] == "Instance":
            if isinstance(type_json.get("args", None), list):
                return (
                    process_typ_str(type_json["type_ref"])
                    + "["
                    + ", ".join(
                        [
                            process_typ_str(self._to_string_type_json(i))
                            for i in type_json["args"]
                        ]
                    )
                    + "]"
                )
            else:
                return process_typ_str(type_json["type_ref"])

        elif type_json[".class"] == "LiteralType":
            return process_typ_str(type_json["fallback"])

        elif type_json[".class"] == "TupleType":
            return (
                "Tuple"
                + "["
                + ", ".join(
                    [
                        process_typ_str(self._to_string_type_json(i))
                        for i in type_json["items"]
                    ]
                )
                + "]"
            )

        elif type_json[".class"] == "UnionType":
            return (
                "Union"
                + "["
                + ", ".join(
                    [
                        process_typ_str(self._to_string_type_json(i))
                        for i in type_json["items"]
                    ]
                )
                + "]"
            )

        else:
            return repr(getattr(types, type_json[".class"]).deserialize(type_json))

    def to_string(self) -> str:
        return self._to_string_type_json(self.type_json)
