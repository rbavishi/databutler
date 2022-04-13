import ast

def is_library_used(code: str, qual_name: str) -> bool:
    c_ast = ast.parse(code)
    print(ast.dump(c_ast))

    vars_to_track = set()
    for node in ast.walk(c_ast):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == qual_name:
                    if alias.asname is None:
                        vars_to_track.add(alias.name)
                    else:
                        vars_to_track.add(alias.asname)

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                full_name = node.module + "." + alias.name
                if full_name == qual_name:
                    vars_to_track.add(alias.asname or full_name)

    for node in ast.walk(c_ast):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in vars_to_track:
            return True

    return False

def are_any_libraries_used(code: str, qual_list: str) -> bool:
    return any([is_library_used(code, qual_name) for qual_name in qual_list])

def are_all_libraries_used(code: str, qual_list: str) -> bool:
    return all([is_library_used(code, qual_name) for qual_name in qual_list])