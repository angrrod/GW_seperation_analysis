from enum import Enum
class Pipeline_type(Enum):
    SINGLE       = "single_likl"
    HIERARCHICAL = "hierarchical"
    JOINT        = "joint_likl"
    TASNET       = "tasNet"

    @property
    def code(self) -> str:
        return self.value

    @property
    def type(self):
        # Lazy import inside property to avoid circular import at module import time
        if self is Pipeline_type.SINGLE:
            from Pipeline.SingleLikelihoodPipeline import SingleLikelihoodPipeline
            return SingleLikelihoodPipeline
        if self is Pipeline_type.HIERARCHICAL:
            from Pipeline.HierarchicalPipeline import HierarchicalPipeline
            return HierarchicalPipeline
        if self is Pipeline_type.JOINT:
            from Pipeline.JointLikelihoodPipeline import JointLikelihoodPipeline
            return JointLikelihoodPipeline
        if self is Pipeline_type.TASNET:
            from Pipeline.TasNetPipeline import TasNetPipeline
            return TasNetPipeline
        raise KeyError(self)