from abc import ABC, abstractmethod

import gradio as gr

from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls


class Widget(ABC):
    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def update(self, dstats: dmt_cls):
        pass

    @property
    @abstractmethod
    def output_components(self):
        pass

    @abstractmethod
    def add_events(self, state: gr.State):
        pass
