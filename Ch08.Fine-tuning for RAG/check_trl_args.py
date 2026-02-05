import inspect
from trl.trainer import SFTTrainer, SFTConfig

print(f"SFTTrainer init parameters: {list(inspect.signature(SFTTrainer.__init__).parameters)}")

