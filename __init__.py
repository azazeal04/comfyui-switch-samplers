from .nodes import *

NODE_CLASS_MAPPINGS = {
    "StepSwitchKSampler": StepSwitchKSampler,
    "MultiStepKSampler": MultiStepKSampler,
    "CrossStepSwitchKSampler": CrossStepSwitchKSampler,
    "CrossMultiStepKSampler": CrossMultiStepKSampler,
}

__all__ = list(NODE_CLASS_MAPPINGS.keys())