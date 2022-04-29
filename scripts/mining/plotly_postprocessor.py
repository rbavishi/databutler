import ast
import yaml
import astunparse
import astpretty
import copy
import shutil

def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

class KeyWordRemover(ast.NodeTransformer):
    def visit_Call(self, node):
        node_new = copy.deepcopy(node)
        if (hasattr(node, 'keywords')):
            keywords = []
            for kw in node.keywords:
                # if it's not setting a title, we're good. Keep it.
                if kw.arg != 'title' and kw.arg != 'title_text' and kw.arg != 'layout_title_text':
                    keywords.append(kw)
            node_new.keywords = keywords
        return node_new

yaml.add_representer(str, str_presenter)

DEEP = 'deep'

# def cleaner(func):
#     ast_code = ast.parse(func['code'])
#     for node in ast.walk(ast_code):
#         for child in ast.iter_child_nodes(node):
#             if (isinstance(child, ast.keyword)):
#                 if (child.arg == 'title'):
#                     cleaned_keywords = [
#                         keyword for keyword in node.keywords
#                             if keyword.arg != 'title'
#                     ]
#                     node.keywords = cleaned_keywords

#                 if (child.arg == 'title_text'):
#                     cleaned_keywords = [
#                         keyword for keyword in node.keywords
#                             if keyword.arg != 'title_text'
#                     ]
#                     node.keywords = cleaned_keywords

#     code = astunparse.unparse(ast_code)
#     func['code'] = code
#     return func

import os
with open('files.txt', 'r') as f:
    for line in f.readlines():
        filepath = line.replace('\n', '')
        file = open(filepath, 'r')

        os.makedirs(f'deep/{filepath[2:-19]}', exist_ok=True)
        outfile = open(f'deep/{filepath[2:]}', 'w+')
        data = yaml.safe_load(file)

        viz_functions = data['viz_functions']
        for viz_function in viz_functions:
            viz_code = viz_function['code']
            viz_ast = ast.parse(viz_code)
            new_viz_ast = ast.fix_missing_locations(KeyWordRemover().visit(viz_ast))
            new_viz_code = astunparse.unparse(new_viz_ast)
            print('Compare:\n')
            print(f'OLD: {viz_code}')
            print(f'NEW: {new_viz_code}')

            # update viz_functions
            viz_function['code'] = new_viz_code

        # write back to file -> data should be updated
        yaml.dump(data, outfile)

