from abc import ABC, abstractmethod
from typing import List

from simpleoptions import BaseOption


class GenericOptionGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_options(self) -> List["BaseOption"]:
        pass

    @abstractmethod
    def train_option(self, option: BaseOption) -> None:
        pass
