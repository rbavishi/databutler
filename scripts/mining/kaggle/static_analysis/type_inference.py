import io
import re
import tempfile
import textwrap
from typing import Dict, Tuple, Optional, Set, Union

import mypy
from mypy.main import main

from databutler.pat import astlib
from databutler.utils import code as codeutils
from databutler.utils.logging import logger


class _TypeInferenceInstrumenter(astlib.AstTransformer):
    def __init__(self):
        super().__init__()
        self._node_to_idx_mapping: Dict[astlib.BaseExpression, int] = {}
        self._cur_ast: Optional[astlib.AstNode] = None
        self._uninstrumentable_exprs: Set[astlib.AstNode] = set()

    def process(self, ast_root: astlib.AstNode) -> Tuple[str, Dict[astlib.BaseExpression, int]]:
        self._node_to_idx_mapping.clear()
        self._cur_ast = ast_root
        self._uninstrumentable_exprs.clear()
        return astlib.to_code(ast_root.visit(self)), self._node_to_idx_mapping.copy()

    def on_visit(self, node: astlib.AstNode):
        if isinstance(node, astlib.ConcatenatedString):
            self._uninstrumentable_exprs.add(node.left)
            self._uninstrumentable_exprs.add(node.right)

        if isinstance(node, astlib.FormattedString):
            return False

        if isinstance(node, astlib.Yield):
            self._uninstrumentable_exprs.add(node)

        if astlib.is_starred_expr(node):
            self._uninstrumentable_exprs.add(node)

        return True

    def on_leave(self, original_node: astlib.AstNode, updated_node: astlib.AstNode):
        if isinstance(original_node, astlib.BaseExpression):
            assert isinstance(updated_node, astlib.BaseExpression)
            if astlib.expr_is_evaluated(original_node, context=self._cur_ast):
                if original_node not in self._uninstrumentable_exprs:
                    return self._process_expr(original_node, updated_node)

        return updated_node

    def _process_expr(
            self, original_node: astlib.BaseExpression, updated_node: astlib.BaseExpression
    ) -> astlib.BaseExpression:
        if original_node not in self._node_to_idx_mapping:
            self._node_to_idx_mapping[original_node] = len(self._node_to_idx_mapping)

        idx = self._node_to_idx_mapping[original_node]
        if isinstance(updated_node, astlib.GeneratorExp) and len(updated_node.lpar) == 0:
            updated_node = astlib.with_changes(updated_node,
                                               lpar=[astlib.cst.LeftParen()],
                                               rpar=[astlib.cst.RightParen()])

        tuple_expr = astlib.create_tuple_expr([
            updated_node,
            self._create_reveal_type_call(astlib.SimpleString(repr(f"_TYPE_IDX_{idx}"))),
            self._create_reveal_type_call(original_node),
        ])
        index_expr = astlib.parse_expr("dummy[2]")
        new_node = astlib.with_changes(index_expr, value=tuple_expr)

        return new_node

    def _create_reveal_type_call(self, expr: astlib.BaseExpression) -> astlib.Call:
        return astlib.Call(func=astlib.create_name_expr("reveal_type"),
                           args=[
                               astlib.Arg(value=expr)
                           ])


def _parse_union_type(t: str) -> Set[str]:
    if t.startswith("Union[") and t.endswith("]"):
        return set(t[len("Union["): -1].split(", "))

    return {t}


def run_mypy(source: Union[str, astlib.AstNode]):
    if isinstance(source, str):
        src_ast = astlib.parse(source)
    elif isinstance(source, astlib.AstNode):
        src_ast = source
    else:
        raise TypeError("Source must be a string or an AST")

    inst_src, node_to_idx = _TypeInferenceInstrumenter().process(src_ast)
    # print(codeutils.normalize_code(inst_src))
    # print(inst_src)
    inferred_types: Dict[astlib.BaseExpression, Set[str]] = {
        # expr: "Any" for expr in node_to_idx.keys()
    }
    idx_to_node = {v: k for k, v in node_to_idx.items()}

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    regex = re.compile('Revealed type is "(.*)"')
    type_idx_regex = re.compile(r'Literal\[\'_TYPE_IDX_([0-9]*)\']')
    with tempfile.NamedTemporaryFile(mode='w') as fp:
        fp.write(codeutils.normalize_code(inst_src))
        fp.flush()
        try:
            main(None, args=[fp.name, "--ignore-missing-imports"], stdout=stdout_buf, stderr=stderr_buf,
                 clean_exit=True)
        except BaseException as e:
            pass

        last_idx: Optional[int] = None
        for line in stdout_buf.getvalue().split("\n"):
            line = line.strip()
            if "Revealed type is" in line:
                try:
                    result = regex.search(line).group(1)
                    if "TYPE_IDX" in line:
                        idx = int(type_idx_regex.search(result).group(1))
                        last_idx = idx
                    elif last_idx is None:
                        # logger.warning(f"Found a type without an existing idx {line}")
                        continue
                    else:
                        inferred_types[idx_to_node[last_idx]] = _parse_union_type(result)
                except:
                    logger.warning(f"Unexpected regex failure for {line}")

    # for node, inferred_type in inferred_types.items():
    #     print(astlib.to_code(node), "   :    ", inferred_type)

    # print(stdout_buf.getvalue())
    # print(stderr_buf.getvalue())
    return src_ast, inferred_types


if __name__ == "__main__":
    code = textwrap.dedent("""
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
train.head()
test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
test.head()
submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
import matplotlib.pyplot as plt
import seaborn as sns
print("Number of Country_Region: ", train['Country_Region'].nunique())
print("Dates are ranging from day", min(train['Date']), "to day", max(train['Date']), ", a total of", train['Date'].nunique(),
      "days")
print("The countries that have Province/Region given are : ", train[train['Province_State'].isna()==False]['Country_Region'].
      unique())
train.columns
print(train.shape)
train['Province_State'].unique()
train1.fillna(0, inplace=True)
show_cum = train.groupby(by='Country_Region')[['ConfirmedCases','Fatalities']].max().reset_index()
plt.figure(figsize=(40,20))
sns.barplot(x='ConfirmedCases',y='Country_Region',data=show_cum[show_cum['ConfirmedCases'] != 0].
            sort_values(by='ConfirmedCases',ascending=False).head(30))
plt.figure(figsize=(20,10))
sns.barplot(x='Fatalities',y='Country_Region',data=show_cum[show_cum['Fatalities'] != 0].
            sort_values(by='Fatalities',ascending=False).head(30))
confirmed_total_dates = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_dates = train.groupby(['Date']).agg({'Fatalities':['sum']})
total_dates = confirmed_total_dates.join(fatalities_total_dates)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
total_dates.plot(ax=ax1)
ax1.set_title("Global confirmed cases", size=13)
ax1.set_ylabel("Total Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
fatalities_total_dates.plot(ax=ax2, color='orange')
ax2.set_title("Global fatalities cases", size=13)
ax2.set_ylabel("Total Number of cases", size=13)
ax2.set_xlabel("Date", size=13)
X_train=train.drop(columns=['Id','ConfirmedCases','Fatalities','Date'])
y_train_cc=train.ConfirmedCases
y_train_ft=train.Fatalities
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
impute=SimpleImputer(strategy='most_frequent')
X_train_1=impute.fit_transform(X_train)
X_train_2=OneHotEncoder().fit_transform(X_train_1)
X_test=test.drop(columns=['ForecastId','Date'])
X_test_1=impute.fit_transform(X_test)
X_test_2=OneHotEncoder().fit_transform(X_test_1)
from sklearn.ensemble import RandomForestRegressor
model_cc=RandomForestRegressor()
model_cc.fit(X_train_2, y_train_cc)
model_cc.score(X_train_2, y_train_cc)
y_pred_cc=model_cc.predict(X_test_2)
y_pred_cc
model_ft=RandomForestRegressor()
model_ft.fit(X_train_2,y_train_ft)
model_ft.score(X_train_2, y_train_ft)
y_pred_ft=model_ft.predict(X_test_2)
y_pred_ft
result=pd.DataFrame({'ForecastId':submission.ForecastId, 'ConfirmedCases':y_pred_cc, 'Fatalities':y_pred_ft})
result.to_csv('/kaggle/working/submission.csv',index=False)
data=pd.read_csv('/kaggle/working/submission.csv')
data.head()
"".join(t for t in a)

def f():
    yield 1
    """)
    run_mypy(code)
