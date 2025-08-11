import abc
from functools import partial

from joblib import Memory
from rich.console import Console
import torch



from viper.configs import config
from viper.src.utils import HiddenPrints

class BaseModel(abc.ABC):
    to_batch = False
    seconds_collect_data = 1.5  # Window of seconds to group inputs, if to_batch is True
    max_batch_size = 10  # Maximum batch size, if to_batch is True. Maximum allowed by OpenAI
    requires_gpu = True
    num_gpus = 1  # Number of required GPUs
    load_order = 0  # Order in which the model is loaded. Lower is first. By default, models are loaded alphabetically

    def __init__(self, gpu_number, device='cuda'):
        self.dev = "mps" if torch.mps.device_count() > torch.cuda.device_count() else "cuda" + f":{gpu_number}" if torch.cuda.is_available() else "cpu"
        self.console = Console(highlight=False)
        self.hp = partial(HiddenPrints, console=self.console, use_newline = config.multiprocessing)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        If to_batch is True, every arg and kwarg will be a list of inputs, and the output should be a list of outputs.
        The way it is implemented in the background, if inputs with defaults are not specified, they will take the
        default value, but still be given as a list to the forward method.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """The name of the model has to be given by the subclass"""
        pass

    @classmethod
    def list_processes(cls):
        """
        A single model can be run in multiple processes, for example if there are different tasks to be done with it.
        If multiple processes are used, override this method to return a list of strings.
        Remember the @classmethod decorator.
        If we specify a list of processes, the self.forward() method has to have a "process_name" parameter that gets
        automatically passed in.
        See GPTModel for an example.
        """
        return [cls.name]