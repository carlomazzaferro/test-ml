
import abc
from typing import List, Type

from testml.types import RunnerT, MetricT, MatrixT, VectorT


class Runner(RunnerT):

    __metaclass__ = abc.ABCMeta

    def __init__(self, model):
        self.model = model
        self.predictions = None

    @abc.abstractmethod
    def run(self, x, labels, metrics, **kwargs):
        ...

    def evaluate(self, labels: VectorT, metrics: List[Type[MetricT]], **kwargs) -> List:
        results = []
        for metric in metrics:
            results.append(metric(labels, self.predictions, **kwargs))
        return results


class KerasRunner(Runner):

    def run(self, x: MatrixT, labels: VectorT, metrics: List[Type[MetricT]], **kwargs) -> List:
        self.predictions = self.model.evaluate(x)
        return self.evaluate(labels, metrics, **kwargs)


class SklearnRunner(Runner):

    def run(self, x: MatrixT, labels: VectorT, metrics: List[Type[MetricT]], **kwargs) -> List:
        self.predictions = self.model.predict(x)
        return self.evaluate(labels, metrics, **kwargs)


