"""
Logic for construction of the hierarchical program trace.
"""
import collections

import attr
from typing import Optional, List, Callable, Any, Dict, Type

from databutler.pat import astlib
from databutler.pat.analysis.clock import LogicalClock
from databutler.pat.analysis.hierarchical_trace.builder_aux import FuncDefLambdaAuxGenerator, ForLoopAuxGenerator
from databutler.pat.analysis.hierarchical_trace.builder_basic import BasicDefEventsGenerator, \
    BasicAccessEventsGenerator, BasicReadEventsGenerator, BasicWriteEventsGenerator, FunctionCallSpecsChecker
from databutler.pat.analysis.hierarchical_trace.builder_utils import TraceItemsCollector, TraceEventsCollector, \
    TraceItemGenerator, TemporaryVariablesGenerator, ValueKeepAliveAgent
from databutler.pat.analysis.hierarchical_trace.core import HierarchicalTrace, TraceItem, TraceEvent
from databutler.pat.analysis.instrumentation import Instrumentation


@attr.s
class HierarchicalTraceInstrumentationHooks:
    """
    Routines to query for live information during execution of the trace instrumentation.
    Allows other instrumentation to peek into the trace instrumentation.
    """
    _clock: LogicalClock = attr.ib()
    _trace_items_collector: TraceItemsCollector = attr.ib()
    _trace_events_collector: TraceEventsCollector = attr.ib()
    _event_handlers: Dict[Type[TraceEvent], List[Callable]] = attr.ib(init=False, default=None)

    def get_latest_item(self, node: astlib.AstNode) -> Optional[TraceItem]:
        """
        Get the latest recorded item for the node.
        :param node:
        :return:
        """
        return self._trace_items_collector.get_last_item_for_node(node)

    def install_event_handler(self, event_type: Type[TraceEvent], handler: Callable):
        if self._event_handlers is None:
            self._event_handlers = collections.defaultdict(list)

        self._event_handlers[event_type].append(handler)
        self._trace_events_collector.install_event_handlers(self._event_handlers)


@attr.s(cmp=False, repr=False)
class HierarchicalTraceInstrumentation(Instrumentation):
    _clock: LogicalClock = attr.ib(init=False, default=None)
    _trace_items_collector: TraceItemsCollector = attr.ib(init=False, default=None)
    _trace_events_collector: TraceEventsCollector = attr.ib(init=False, default=None)
    _hooks: HierarchicalTraceInstrumentationHooks = attr.ib(init=False, default=None)

    _finalized: bool = attr.ib(init=False, default=None)
    _value_keep_alive_agent: ValueKeepAliveAgent = attr.ib(init=False, default=None)

    def preprocess(self, ast_root: astlib.AstNode):
        super().preprocess(ast_root)
        self._finalized = False

    @classmethod
    def build(cls, clock: LogicalClock):
        trace_items_collector = TraceItemsCollector()
        trace_events_collector = TraceEventsCollector()
        temp_vars_generator = TemporaryVariablesGenerator()

        value_keep_alive_agent = ValueKeepAliveAgent()

        trace_items_gen = TraceItemGenerator(clock=clock, collector=trace_items_collector)

        access_events_gen = BasicAccessEventsGenerator(clock=clock,
                                                       trace_events_collector=trace_events_collector,
                                                       trace_items_collector=trace_items_collector)
        basic_def_events_gen = BasicDefEventsGenerator(clock=clock,
                                                       trace_events_collector=trace_events_collector,
                                                       trace_items_collector=trace_items_collector)
        basic_read_events_gen = BasicReadEventsGenerator(clock=clock,
                                                         trace_events_collector=trace_events_collector,
                                                         trace_items_collector=trace_items_collector)
        basic_write_events_gen = BasicWriteEventsGenerator(clock=clock,
                                                           trace_events_collector=trace_events_collector,
                                                           trace_items_collector=trace_items_collector)
        func_specs_checker = FunctionCallSpecsChecker(clock=clock,
                                                      trace_events_collector=trace_events_collector,
                                                      trace_items_collector=trace_items_collector)

        func_lambda_aux_gen = FuncDefLambdaAuxGenerator(clock=clock,
                                                        trace_events_collector=trace_events_collector,
                                                        trace_items_collector=trace_items_collector,
                                                        temp_vars_generator=temp_vars_generator)
        for_loop_aux_gen = ForLoopAuxGenerator(clock=clock,
                                               trace_events_collector=trace_events_collector,
                                               trace_items_collector=trace_items_collector,
                                               temp_vars_generator=temp_vars_generator)

        result = cls.from_generators(value_keep_alive_agent,
                                     trace_items_gen,
                                     basic_def_events_gen, access_events_gen,
                                     basic_read_events_gen, basic_write_events_gen,
                                     func_specs_checker,
                                     func_lambda_aux_gen, for_loop_aux_gen)
        result._clock = clock
        result._trace_items_collector = trace_items_collector
        result._trace_events_collector = trace_events_collector
        result._hooks = HierarchicalTraceInstrumentationHooks(clock=clock,
                                                              trace_items_collector=trace_items_collector,
                                                              trace_events_collector=trace_events_collector)

        #  We store this in order to clean up later.
        result._value_keep_alive_agent = value_keep_alive_agent

        return result

    def finalize(self):
        """
        It is recommended to call this after instrumented code has been run in order to clean up.
        :return:
        """
        if not self._finalized:
            self._value_keep_alive_agent.cleanup()
            self._finalized = True

    def get_hierarchical_trace(self) -> HierarchicalTrace:
        return HierarchicalTrace(
            events=self._trace_events_collector.get_events(),
            items=self._trace_items_collector.get_items(),
        )

    def get_hooks(self):
        return self._hooks


def get_hierarchical_trace_instrumentation(clock: Optional[LogicalClock] = None):
    if clock is None:
        clock = LogicalClock()

    return HierarchicalTraceInstrumentation.build(clock=clock)
