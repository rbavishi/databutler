import textwrap
from typing import Dict, List, Tuple

import attrs

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.generic.corpus.code_changes import change

_CodeChangeDescT = str


@attrs.define(eq=False, repr=False)
class CorpusItem:
    #  The base datana function.
    func: DatanaFunction
    #  Description for the vanilla version of the function.
    #  That is, with all the (code-removing) changes applied.
    vanilla_desc: str
    #  A dictionary mapping change IDs to tuples of changes and their descriptions that can be applied on the function
    #  code to obtain variants.
    change_dict: Dict[str, Tuple[change.BaseCodeRemovalChange, _CodeChangeDescT]]
    #  The default set of changes to apply. Note that because changes are based on removal, the descriptions are in
    #  the opposite direction. That is, they describe the change when the code is added. So selecting a change means
    #  *not* applying it, while the ones that are not selected, need to be applied.
    #  The field below corresponds to the set of changes that are *not* selected.
    default_changeset: List[str]
    #  A unique identifier
    uid: str

    def apply_changes(self, change_ids: List[str]) -> str:
        changes = [self.change_dict[i][0] for i in change_ids]
        if len(changes) == 0:
            return self.func.code_str

        return changes[0].__class__.apply_changes(self.func.code_str, changes)


class PieChartCorpus:
    #  ------------------------
    #  First VizSmith function
    #  ------------------------

    BASE_CODE_1 = textwrap.dedent("""
    def visualization(df, col0):
        import matplotlib.pyplot as plt
        counts = df[col0].value_counts()
        plt.pie(counts, labels=counts.index, startangle=90, shadow=True)
    """)

    CHANGE_1_1 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Arg", index=1)],  # Corresponds to labels=counts
        children=[],
    )

    DESC_1_1 = "Add labels for the wedges"

    CHANGE_2_1 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Arg", index=2)],  # Corresponds to startangle=90
        children=[],
    )

    DESC_2_1 = "Change the starting angle to 90 degrees"

    CHANGE_3_1 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Arg", index=3)],  # Corresponds to shadow=True
        children=[],
    )

    DESC_3_1 = "Add shadows for the wedges"

    FUNC_1 = DatanaFunction(
        code_str=BASE_CODE_1,
        func_name="visualization",
        pos_args=[None, None],
        kw_args={},
        uid="pie-chart-1",
    )

    VANILLA_DESC_1 = "Draw a pie-chart of [COL0]"

    CORPUS_ITEM_1 = CorpusItem(
        func=FUNC_1,
        vanilla_desc=VANILLA_DESC_1,
        change_dict={
            "c1": (CHANGE_1_1, DESC_1_1),
            "c2": (CHANGE_2_1, DESC_2_1),
            "c3": (CHANGE_3_1, DESC_3_1),
        },
        default_changeset=["c1", "c2", "c3"],
        uid=FUNC_1.uid,
    )

    #  ------------------------
    #  Second VizSmith function
    #  ------------------------

    BASE_CODE_2 = textwrap.dedent("""
    def visualization(df, col0):
        import matplotlib.pyplot as plt
        plt.style.use('fivethirtyeight')
        counts = df[col0].value_counts()
        plt.pie(counts, labels=counts.index, startangle=180)
        plt.title("Pie-Chart")
    """)

    CHANGE_1_2 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Arg", index=2)],  # Corresponds to labels=counts
        children=[],
    )

    DESC_1_2 = "Add labels for the wedges"

    CHANGE_2_2 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Arg", index=3)],  # Corresponds to startangle=180
        children=[],
    )

    DESC_2_2 = "Change the starting angle to 180 degrees"

    CHANGE_3_2 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Expr", index=2)],  # Corresponds to plt.title
        children=[],
    )

    DESC_3_2 = "Add 'Pie-Chart' as the title"

    CHANGE_4_2 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Expr", index=0)],  # Corresponds to plt.style
        children=[],
    )

    DESC_4_2 = "Use the Five-Thirty-Eight style"

    FUNC_2 = DatanaFunction(
        code_str=BASE_CODE_2,
        func_name="visualization",
        pos_args=[None, None],
        kw_args={},
        uid="pie-chart-2",
    )

    VANILLA_DESC_2 = "Draw a pie-chart of [COL0]"

    CORPUS_ITEM_2 = CorpusItem(
        func=FUNC_2,
        vanilla_desc=VANILLA_DESC_2,
        change_dict={
            "c1": (CHANGE_1_2, DESC_1_2),
            "c2": (CHANGE_2_2, DESC_2_2),
            "c3": (CHANGE_3_2, DESC_3_2),
            "c4": (CHANGE_4_2, DESC_4_2),
        },
        default_changeset=["c1", "c2", "c3", "c4"],
        uid=FUNC_2.uid,
    )

    @classmethod
    def get_corpus_elems(cls) -> List[CorpusItem]:
        return [cls.CORPUS_ITEM_1, cls.CORPUS_ITEM_2]


class DistPlotCorpus:
    #  ------------------------
    #  First VizSmith function
    #  ------------------------

    BASE_CODE_1 = textwrap.dedent("""
    def visualization(df, col0):
        import seaborn as sns
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        sns.distplot(df[col0], kde=False, fit=norm)
        plt.title("Distribution Plot")
    """)

    CHANGE_1_1 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Arg", index=1)],  # Corresponds to kde=False
        children=[],
    )

    DESC_1_1 = "Remove the kernel density estimate"

    CHANGE_2_1 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Arg", index=2)],  # Corresponds to fit=norm
        children=[],
    )

    DESC_2_1 = "Add a maximum likelihood gaussian distribution fit"

    CHANGE_3_1 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Expr", index=1)],  # Corresponds to plt.title
        children=[],
    )

    DESC_3_1 = "Add 'Distribution Plot' as the title"

    FUNC_1 = DatanaFunction(
        code_str=BASE_CODE_1,
        func_name="visualization",
        pos_args=[None, None],
        kw_args={},
        uid="dist-chart-1",
    )

    VANILLA_DESC_1 = "Visualize [COL0] as a histogram"

    CORPUS_ITEM_1 = CorpusItem(
        func=FUNC_1,
        vanilla_desc=VANILLA_DESC_1,
        change_dict={
            "c1": (CHANGE_1_1, DESC_1_1),
            "c2": (CHANGE_2_1, DESC_2_1),
            "c3": (CHANGE_3_1, DESC_3_1),
        },
        default_changeset=["c1", "c2", "c3"],
        uid=FUNC_1.uid,
    )

    BASE_CODE_2 = textwrap.dedent("""
    def visualization(df, col0):
        import seaborn as sns
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        sns.set_style('darkgrid')
        sns.distplot(df[col0], kde=False, fit=norm)
        plt.title("Distribution Plot")
    """)

    CHANGE_1_2 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Arg", index=2)],  # Corresponds to kde=False
        children=[],
    )

    DESC_1_2 = "Remove the kernel density estimate"

    CHANGE_2_2 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Arg", index=3)],  # Corresponds to fit=norm
        children=[],
    )

    DESC_2_2 = "Add a maximum likelihood gaussian distribution fit"

    CHANGE_3_2 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Expr", index=2)],  # Corresponds to plt.title
        children=[],
    )

    DESC_3_2 = "Add 'Distribution Plot' as the title"

    CHANGE_4_2 = change.SimpleAstLibRemovalChange(
        node_refs=[change.SimpleAstLibNodeRef(node_type="Expr", index=0)],  # Corresponds to plt.style
        children=[],
    )

    DESC_4_2 = "Use a dark-background style with grids"

    FUNC_2 = DatanaFunction(
        code_str=BASE_CODE_2,
        func_name="visualization",
        pos_args=[None, None],
        kw_args={},
        uid="dist-chart-2",
    )

    VANILLA_DESC_2 = "Visualize [COL0] as a histogram"

    CORPUS_ITEM_2 = CorpusItem(
        func=FUNC_2,
        vanilla_desc=VANILLA_DESC_1,
        change_dict={
            "c1": (CHANGE_1_2, DESC_1_2),
            "c2": (CHANGE_2_2, DESC_2_2),
            "c3": (CHANGE_3_2, DESC_3_2),
            "c4": (CHANGE_4_2, DESC_4_2),
        },
        default_changeset=["c1", "c2", "c3", "c4"],
        uid=FUNC_2.uid,
    )

    @classmethod
    def get_corpus_elems(cls) -> List[CorpusItem]:
        return [cls.CORPUS_ITEM_1, cls.CORPUS_ITEM_2]


CORPUS: List[CorpusItem] = [
    *PieChartCorpus.get_corpus_elems(),
    *DistPlotCorpus.get_corpus_elems(),
]
