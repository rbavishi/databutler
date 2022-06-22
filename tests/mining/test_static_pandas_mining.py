import textwrap
import unittest

import databutler.mining.kaggle.static_analysis.pandas_mining_utils
from databutler.mining.kaggle.static_analysis import pandas_mining as mining
from databutler.pat import astlib


class StaticPandasMiningTests(unittest.TestCase):
    def test_find_library_usages_1(self):
        code = textwrap.dedent(
            """
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy
        from random import shuffle
        import random
        
        a = pd.read_csv("titanic.csv")
        plt.plot(a)
        print(numpy.array(a["Survived"]))
        b = [1, 2, 3, 4]
        random.shuffle(b)
        shuffle(b)
        """
        )

        code_ast = astlib.parse(code)
        result = databutler.mining.kaggle.static_analysis.pandas_mining_utils.find_library_usages(
            code_ast
        )
        print(result)
        self.assertEquals(5, len(result))

    def test_find_constants_1(self):
        code = textwrap.dedent(
            """
        a = 20
        b = a + 10
        a = 30
        c = a + 50
        d = {1: 3, 4: 5}
        e = {1, 2, 3}
        f = [1, 2, 3]
        g = (1, 2, 3)
        h = d[1] + next(iter(e)) + f[0] + g[0]
        """
        )

        code_ast = astlib.parse(code)
        result = (
            databutler.mining.kaggle.static_analysis.pandas_mining_utils.find_constants(
                code_ast
            )
        )
        self.assertEqual(6, len(result))
        self.assertSetEqual(
            {int, dict, set, list, tuple}, {type(i) for i in result.values()}
        )

    def test_mine_code_1(self):
        code = textwrap.dedent(
            """
        import pandas as pd
        data1 = pd.read_csv("titanic.csv")
        data2 = pd.read_csv("titanic_test.csv")
        columns = ["Survived", "PClass"]
        a = data1[columns]
        b = data1.AColumn
        c = data1.groupby(["Col1", "Col2"]).AColumn.mean()
        a = b.str.split()
        pd.concat([data1, data2, data1])
        data1.fillna(0, inplace=True)
        data2 = data2.fillna(inplace=False, value=0.0)
        data2 = data2.fillna("0", inplace=False)
        data2 = data2.fillna(len(data2), inplace=False)
        data2 = data2.replace(lambda x: x)
        df = pd.read_csv("unittesting.csv")
        df['State'].apply((lambda state: (state[0] == 'W')))
        df[df['state_name'].isin(('Florida', 'Texas', 'Louisiana', 'Alabama'))]

        df.iloc[0]
        df.iloc[:, 0]
        df.iloc[:, 1:3]
        df.iloc[:, [1, 2]]
        df.iloc[:, lambda d: [0, 2]]
        df.iloc[:, lambda d: 1]
        
        df.loc['0']
        df.loc[:, '0']
        df.loc[:, 'a':'b']
        df.loc[['col1', 'col2']]
        df.loc[:, ['col1', 'col2']]
        df.loc[:, lambda d: ['col1', 'col2']]
        df.loc[:, lambda d: 'a']
        
        df.at[0, 'a']
        df.iat[0, 0]
        """
        )

        for res in mining.mine_code(code):
            print(res)

    def test_mine_code_2(self):
        code = textwrap.dedent(
            """
        import pandas as pd
        import seaborn as sns
        df = pd.read_csv("titanic.csv")
        df1 = pd.read_csv("test.csv")
        df1[df.Survived == 0]
        a = df["Survived"]
        a.str.count("a")
        df.isnull().any()
        df['Address'].str.split('_', expand=True)
        df.add_prefix("col_")
        df['Address'].str.split(' ').str[-1]
        sns.distplot(df['Age'])
        df.loc[:, 'Age'].hist(bins=20)
        """
        )

        for res in mining.mine_code(code):
            print(res)
