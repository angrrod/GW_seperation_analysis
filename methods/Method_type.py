from enum import Enum
class Method_type(Enum):
    SINGLE       = "single_likl"
    HIERARCHICAL = "hierarchical"
    JOINT        = "joint_likl"
    TASNET       = "tasNet"

    @property
    def code(self) -> str:
        return self.value

    @property
    def method(self):
        # Lazy import inside property to avoid circular import at module import time
        if self is Method_type.SINGLE:
            from .SingleSignalMethod import SingleSignalMethod
            return SingleSignalMethod
        if self is Method_type.HIERARCHICAL:
            from .HyrarchicalMethod import HyrarchicalMethod
            return HyrarchicalMethod
        if self is Method_type.JOINT:
            from .JointLikelihoodlMethod import JointLikelihoodlMethod
            return JointLikelihoodlMethod
        if self is Method_type.TASNET:
            from .TasNetMethod import TasNetMethod
            return TasNetMethod
        raise KeyError(self)