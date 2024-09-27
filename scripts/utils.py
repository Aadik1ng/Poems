import os
import yaml
from zenml.steps import step
from typing import Dict, Tuple, Annotated

@step
def load_config() -> Annotated[Dict, "Configuration Dictionary"]:
    with open('../config/config.yaml', 'r') as f:
        return yaml.safe_load(f)
