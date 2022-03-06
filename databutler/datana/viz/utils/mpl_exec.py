from typing import Dict, Any, Optional, Callable, List

from pebble import concurrent


def run_viz_code_matplotlib(code: str,
                            pos_args: List[Any],
                            kw_args: Dict[str, Any],
                            func_name: str = 'visualization',
                            other_globals: Optional[Dict] = None,
                            disable_seaborn_randomization: bool = True,
                            serializer: Callable[['plt.Figure'], Any] = None) -> 'plt.Figure':
    #  Putting imports inside so we can delay the imports as long as possible.
    #  This is especially helpful when using libversioning to temporarily modify sys.path
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        plt.style.use('default')
    except:
        pass
    try:
        mpl.rcParams.update(mpl.rcParamsDefault)
    except:
        pass

    if disable_seaborn_randomization:
        try:
            orig_random_seed_fn = sns.algorithms._handle_random_seed

            def wrapper(_):
                return orig_random_seed_fn(0)

            sns.algorithms._handle_random_seed = wrapper
        except:
            pass

    try:
        orig_fig = plt.figure()
        m = {}
        if other_globals is not None:
            m.update(other_globals)

        exec(code, m, m)
        func = m[func_name]
        func(*pos_args, **kw_args)
        fig = plt.gcf()
        if fig is not orig_fig:
            plt.close(orig_fig)

        if serializer is not None:
            result = serializer(fig)
            plt.close(fig)
            return result

        return fig

    finally:
        try:
            plt.style.use('default')
        except:
            pass
        try:
            mpl.rcParams.update(mpl.rcParamsDefault)
        except:
            pass

        if disable_seaborn_randomization:
            try:
                sns.algorithms._handle_random_seed = orig_random_seed_fn
            except:
                pass


def run_viz_code_matplotlib_mp(code: str,
                               pos_args: List[Any],
                               kw_args: Dict[str, Any],
                               func_name: str = 'visualization',
                               other_globals: Optional[Dict] = None,
                               disable_seaborn_randomization: bool = True,
                               serializer: Callable[['plt.Figure'], Any] = None,
                               timeout: Optional[int] = None):
    func = concurrent.process(timeout=timeout)(run_viz_code_matplotlib)
    future = func(code=code, pos_args=pos_args, kw_args=kw_args, func_name=func_name, other_globals=other_globals,
                  disable_seaborn_randomization=disable_seaborn_randomization, serializer=serializer)

    try:
        result = future.result()
        return result

    except TimeoutError:
        return None

    except Exception as e:
        print(e)
        return None
