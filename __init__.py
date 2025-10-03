from .nodes import *

NODE_CLASS_MAPPINGS = {
    "StepSwitchKSampler": StepSwitchKSampler,
    "MultiStepKSampler": MultiStepKSampler,
}

__all__ = list(NODE_CLASS_MAPPINGS.keys())