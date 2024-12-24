from math import ceil
from abc import abstractmethod
from pathlib import Path
from functools import cached_property, partial, reduce
from typing import (
    Any,
    TypeVar,
    Generic,
    Iterable,
    Callable,
    Sequence
)
from concurrent.futures import ThreadPoolExecutor
from mully.core.enums import TaskType
from mully.core.types import (
    EvaluationRequest,
    BaseDataset,
    PROVIDER,
)

P = TypeVar("P")
T = TypeVar("T", bound=EvaluationRequest)



class BaseBurfaModel(Generic[P]):
    """_summary_: Interface for All Models"""
    def __init__(self, provider:PROVIDER, task:TaskType):
        self.provider = provider
        self.task = task

    @abstractmethod
    @cached_property
    def model(self) -> P:
        """_summary_: Model"""
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def predict(self, data:type[BaseDataset]) -> list[Any]:
        """_summary_: Predict"""
        raise NotImplementedError("Method not implemented")


class BaseAgent(Generic[T, P]):
    """_summary_: Interface for All Worker Agents"""
    def __init__(
        self,
        request: T,
        iteration: int,
        completed:int=0
    ):
        self.request = request
        self.iteration = iteration
        self.completed = completed #TODO: Fixme

    @abstractmethod
    @cached_property
    def model(self) -> BaseBurfaModel[P]:
        """_summary_: Model"""
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    @cached_property
    def total_tasks(self) -> int:
        """_summary_: Progress"""
        raise NotImplementedError("Method not implemented")

    def progress(self) -> None:
        """_summary_: Progress"""
        return round((self.completed / self.total_tasks) * 100, 2)

    @abstractmethod
    def execute(self) -> P:
        """_summary_: Execute"""
        raise NotImplementedError("Method not implemented")


class BaseWorker(Generic[T, P]):
    """_summary_: Interface for Runner"""

    def __init__(self, request: T, iterations: int|None=5):
        self.request = request
        self.iterations = iterations

    @cached_property
    def workload(self) -> int:
        """_summary_: Workload"""
        return ceil(self.request.dataset_sample_size / 100)

    @abstractmethod
    def agent(self, provider: PROVIDER, iteration: int) -> BaseAgent[T, P]:
        """_summary_: Agent"""
        raise NotImplementedError("Method not implemented")

    def _execute(self, provider: PROVIDER, iteration: int) -> P:
        """_summary_: Kickoff"""
        return self.agent(provider, iteration).execute()

    def execute(self) -> Iterable[Iterable[P]]:
        """_summary_: Execute the agent."""
        #total, available = mp.cpu_count(), os.sched_getaffinity(0) #TODO: Fixme
        with ThreadPoolExecutor(max_workers=self.workload) as executors:
            results = executors.map(self._execute, range(self.workload))
            executors.shutdown(wait=True)
        return results


class BasePipeline(Generic[T, P]):
    """_summary_: Pipeline class for sequential functions"""

    def __init__(self, state: T | None = None):
        """_summary_: Initialize pipeline"""
        self.state =state
        self.functions: Sequence[Callable[[Any], Any]] = []

    def next(self, function: Callable, *args, **kwargs):
        """_summary_: Add function to pipeline"""
        self.functions += [partial(function, *args, **kwargs)]

    def _init_pipeline(self, state: T, *args, **kwargs):
        """_summary_: Initialize This Pipeline"""
        super(BasePipeline, self).__init__(state)
        
        log_path = Path(self.state.log_dir)
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=True)
        self.next(self.start, *args, **kwargs)

    def run(self, inputs: T) -> P:
        """_summary_: Kickoff pipeline"""
        self._init_pipeline(inputs)
        return reduce(lambda _this, _next: _next(_this), self.functions)

    @abstractmethod
    def start(self, *args, **kwargs):
        """_summary_: Start pipeline"""
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def end(self, pipeline_output: P) -> P:
        """_summary_: Start pipeline"""
        raise NotImplementedError("Method not implemented")
