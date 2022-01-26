import collections
import textwrap
import unittest
from typing import List, Dict

import attr
import pandas as pd

from databutler.pat import astlib
from databutler.pat.analysis.clock import LogicalClock
from databutler.pat.analysis.hierarchical_trace.builder import get_hierarchical_trace_instrumentation
from databutler.pat.analysis.hierarchical_trace.core import DefEvent, AccessEvent, ObjWriteEvent
from databutler.pat.analysis.instrumentation import Instrumenter, ExprWrappersGenerator, ExprWrapper, Instrumentation


@attr.s(cmp=False, repr=False)
class DfRecorder(ExprWrappersGenerator):
    _dfs = attr.ib(init=False, factory=list)

    def gen_expr_wrappers(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
        wrappers = {}
        for expr in self.iter_valid_exprs(ast_root):
            wrappers[expr] = [ExprWrapper(callable=self.record_df, name=self.gen_wrapper_id())]
        return wrappers

    def record_df(self, value):
        if isinstance(value, pd.DataFrame):
            self._dfs.append(value)

        return value

    def get_recorded_dfs(self):
        return self._dfs[:]


class HierarchicalTraceTests(unittest.TestCase):
    def get_trace(self, code):
        code_ast = astlib.parse(code)
        clock = LogicalClock()
        instrumentation = get_hierarchical_trace_instrumentation(clock=clock)
        instrumenter = Instrumenter(instrumentation)
        new_ast, globs = instrumenter.process(code_ast)
        new_code = astlib.to_code(new_ast)
        exec(new_code, globs, globs)

        return instrumentation.get_hierarchical_trace()

    def test_1(self):
        code = """
        def func(x, y, z, *args, a=10, b=100, c=200, default=20, **kwargs):
            return x + y + z + a

        func(10, 15, *[20, 30], 40, *(i for i in (50, 60)), **{'c': 150, 'b': 200, 'e': 9}, a=300, d=500) 
        """
        code = textwrap.dedent(code).strip()

        trace = self.get_trace(code)
        items = trace.get_items()
        for i in items:
            if i.is_auxiliary_type():
                print(i.start_time, i.end_time, astlib.to_code(i.ast_node))

        events = trace.get_events()
        def_events = [d for d in events if isinstance(d, DefEvent)]
        access_events = [a for a in events if isinstance(a, AccessEvent)]
        for param in ['x', 'y', 'z', 'args', 'a', 'b', 'c', 'default', 'kwargs']:
            valid_defs = [d for d in def_events if d.name == param]
            self.assertEqual(1, len(valid_defs), param)
            self.assertIsNotNone(valid_defs[0].owner, param)

        self.assertTrue(any(d.name == 'func' for d in def_events))
        self.assertTrue(any(a.name == 'func' and a.def_event is not None for a in access_events))

    def test_2(self):
        code = """
        import pandas as pd
        df = pd.DataFrame([
        ["Pants", 50, 70], 
        ["Pants", 100, 90], 
        ["Shirts", 80, 110]], columns=["Type", "Low", "High"])

        df.Low = df.Low.apply(lambda x: x * 2)
        """
        code = textwrap.dedent(code).strip()

        events = self.get_trace(code).get_events()
        def_events = [d for d in events if isinstance(d, DefEvent)]
        self.assertEqual(3, len([d for d in def_events if d.name == 'x']))
        for d in def_events:
            if d.name == 'x':
                self.assertIsNone(d.owner)

    def test_3(self):
        code = """
        import pandas as pd
        df = pd.DataFrame([
        ["Pants", 50, 70], 
        ["Pants", 100, 90], 
        ["Shirts", 80, 110]], columns=["Type", "Low", "High"])
        
        s = 0
        for c1, c2 in zip(df.Low, df.High):
            s += (c1 + c2) / 2
            
        df.Low = df.Low + s
        """
        code = textwrap.dedent(code).strip()

        ctr_dict = collections.defaultdict(int)
        trace = self.get_trace(code)
        items = trace.get_items()
        events = trace.get_events()
        for i in items:
            if i.is_auxiliary_type():
                print(i.start_time, i.end_time, astlib.to_code(i.ast_node))

        def_events = [d for d in events if isinstance(d, DefEvent)]
        for d in def_events:
            ctr_dict[d.name] += 1

        self.assertEqual(3, ctr_dict['c1'])
        self.assertEqual(3, ctr_dict['c2'])

    def test_4(self):
        code = """
        a = 2
        a += 3
        [a, b] = [6, 7]
        """
        code = textwrap.dedent(code).strip()

        events = self.get_trace(code).get_events()
        self.assertTrue(any(isinstance(e, AccessEvent) for e in events))
        access_event = next(e for e in events if isinstance(e, AccessEvent))
        self.assertEqual("a = 2", astlib.to_code(access_event.def_event.owner.ast_node))

    def test_5(self):
        code = """
        a = [1, 2, 3, 4]
        a.extend([1,2]) 
        """
        code = textwrap.dedent(code).strip()

        events = self.get_trace(code).get_events()
        self.assertTrue(any(isinstance(e, ObjWriteEvent) for e in events))

    def test_6(self):
        code = """
        import pandas as pd
        df2 = pd.DataFrame([
        ["Pants", 50, 70], 
        ["Shirts", 80, 110]], columns=["Type", "Low", "High"])
        df = pd.DataFrame([
        ["Pants", 50, 70], 
        ["Pants", 100, 90], 
        ["Shirts", 80, 110]], columns=["Type", "Low", "High"])
        df3 = df + df

        df3['Low'] = df3['Low'].apply(lambda x: x * 2) 
        """
        code = textwrap.dedent(code).strip()
        code_ast = astlib.parse(code)
        clock = LogicalClock()
        df_recorder = DfRecorder()
        trace_instrumentation = get_hierarchical_trace_instrumentation(clock=clock)
        instrumentation = trace_instrumentation | Instrumentation.from_generators(df_recorder)

        instrumenter = Instrumenter(instrumentation)
        new_ast, globs = instrumenter.process(code_ast)
        new_code = astlib.to_code(new_ast)
        exec(new_code, globs, globs)

        trace = trace_instrumentation.get_hierarchical_trace()
        obj_ids = trace.get_objs_item_depends_on(trace.items[-1])
        for df in df_recorder.get_recorded_dfs():
            if df.shape[0] == 2:
                self.assertNotIn(id(df), obj_ids)
            elif df.shape[0] == 3:
                self.assertIn(id(df), obj_ids)

    def test_7(self):
        code = """
        a = 2
        a += 3
        [a, b] = [6, 7]
        """
        code = textwrap.dedent(code).strip()
        code_ast = astlib.parse(code)

        cnt = 0

        def inc(event, **kwargs):
            nonlocal cnt
            cnt += 1

        clock = LogicalClock()
        trace_instrumentation = get_hierarchical_trace_instrumentation(clock=clock)
        hooks = trace_instrumentation.get_hooks()
        hooks.install_event_handler(DefEvent, inc)

        instrumenter = Instrumenter(trace_instrumentation)
        new_ast, globs = instrumenter.process(code_ast)
        new_code = astlib.to_code(new_ast)
        exec(new_code, globs, globs)
        self.assertEqual(4, cnt)

    def test_stress_test_1(self):
        code = """
        import os
        import pandas as pd
        from numpy import nan, ndarray
        import matplotlib.pyplot
        from textwrap import *

        matplotlib.pyplot

        def func(x, y, z, *args, a=10, b=100, c=200, default=20, **kwargs):
            return x + y + z + a

        func(10, *[20, 30], 40, 50, **{'c': 150, 'b': 200}, a=300, d=500)
        a = pd.DataFrame({1: [2,3]})
        b = [1,2,3,4]
        b.append(2)

        w = [i for i in [1,2,3,4]]
        print(w)

        for m in [10, 20]:
            print(m)

        with open('trial.txt', 'w') as f:
            pass
            
        os.unlink('trial.txt')
        """
        code = textwrap.dedent(code).strip()
        code_ast = astlib.parse(code)
        clock = LogicalClock()
        instrumentation = get_hierarchical_trace_instrumentation(clock=clock)
        instrumenter = Instrumenter(instrumentation)
        new_ast, globs = instrumenter.process(code_ast)
        new_code = astlib.to_code(new_ast)
        exec(new_code, globs, globs)

        events = instrumentation.get_hierarchical_trace().get_events()
        self.assertGreater(len(events), 0)

    def test_stress_test_2(self):
        code = """
    import pandas as pd
    import numpy as np

    pd.plotting.register_matplotlib_converters()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('dark')
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    print('Setup complete')
    # Load and display train data
    train_data = pd.read_csv("https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/train.csv")
    test_data = pd.read_csv("https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/test.csv")
    train_data.head()
    # Load and display test data
    test_data.head()
    train_data.info()
    test_data.info()
    train_data.Fare.describe()
    fare_ranges = pd.qcut(train_data.Fare, 4, labels=['Low', 'Mid', 'High', 'Very high'])

    def remove_zero_fares(row):
        if row.Fare == 0:
            row.Fare = np.NaN
        return row


    # Apply the function
    print('Number of zero-Fares: {:d}'.format(train_data.loc[train_data.Fare == 0].shape[0]))
    print('Number of zero-Fares: {:d}'.format(test_data.loc[test_data.Fare == 0].shape[0]))
    train_data = train_data.apply(remove_zero_fares, axis=1)
    test_data = test_data.apply(remove_zero_fares, axis=1)
    # Check if it did the job
    print('Number of zero-Fares: {:d}'.format(train_data.loc[train_data.Fare == 0].shape[0]))
    print('Number of zero-Fares: {:d}'.format(test_data.loc[test_data.Fare == 0].shape[0]))
    train_data['Title'] = train_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    test_data['Title'] = test_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    train_data['Title'].value_counts()
    test_data['Title'].value_counts()
    # Substitute rare female titles
    train_data['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)
    test_data['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)
    # Substitute rare male titles
    train_data['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
    test_data['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
    train_data.groupby('Title').Survived.mean()
    # Extract the first two letters
    train_data['Ticket_lett'] = train_data.Ticket.apply(lambda x: x[:2])
    test_data['Ticket_lett'] = test_data.Ticket.apply(lambda x: x[:2])
    # Calculate ticket length
    train_data['Ticket_len'] = train_data.Ticket.apply(lambda x: len(x))
    test_data['Ticket_len'] = test_data.Ticket.apply(lambda x: len(x))
    # Creation of a new Fam_size column
    train_data['Fam_size'] = train_data['SibSp'] + train_data['Parch'] + 1
    test_data['Fam_size'] = test_data['SibSp'] + test_data['Parch'] + 1
    # Creation of four groups
    train_data['Fam_type'] = pd.cut(train_data.Fam_size, [0, 1, 4, 7, 11], labels=['Solo', 'Small', 'Big', 'Very big'])
    test_data['Fam_type'] = pd.cut(test_data.Fam_size, [0, 1, 4, 7, 11], labels=['Solo', 'Small', 'Big', 'Very big'])
    y = train_data['Survived']
    features = ['Pclass', 'Fare', 'Title', 'Embarked', 'Fam_type', 'Ticket_len', 'Ticket_lett']
    X = train_data[features]
    X.head()
        """
        code = textwrap.dedent(code).strip()
        code_ast = astlib.parse(code)
        clock = LogicalClock()
        instrumentation = get_hierarchical_trace_instrumentation(clock=clock)
        instrumenter = Instrumenter(instrumentation)
        new_ast, globs = instrumenter.process(code_ast)
        new_code = astlib.to_code(new_ast)
        exec(new_code, globs, globs)

        events = instrumentation.get_hierarchical_trace().get_events()
        self.assertGreater(len(events), 0)
