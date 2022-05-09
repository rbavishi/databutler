from typing import Dict, Any, Union

import attrs
from mypy.types import Type as MypyType


@attrs.define
class SerializedMypyType:
    type_json: Union[str, Dict[str, Any]]

    @classmethod
    def from_mypy_type(cls, mypy_type: MypyType) -> 'SerializedMypyType':
        return SerializedMypyType(mypy_type.serialize())

    def is_literal(self) -> bool:
        if isinstance(self.type_json, str):
            return False

        t_json = self.type_json
        if t_json['.class'] == 'LiteralType':
            return True

        if t_json['.class'] == 'Instance' and t_json.get('last_known_value', None) is not None:
            return True

        return False

    def get_literal_value(self) -> Union[str, int, bool]:
        assert isinstance(self.type_json, dict)
        if self.type_json['.class'] == 'LiteralType':
            return self.type_json['value']

        if self.type_json['.class'] == 'Instance':
            return self.type_json['last_known_value']['value']

        raise ValueError(f"Cannot extract literal value from {self.type_json}")

    def is_module_type(self) -> bool:
        if isinstance(self.type_json, str) and self.type_json == "types.ModuleType":
            return True

        if isinstance(self.type_json, dict) and self.type_json['.class'] == 'AnyType':
            if self.type_json.get('type_of_any', None) == 3:
                return True
            if self.type_json.get('missing_import_name', None) is not None:
                return True

        return False
