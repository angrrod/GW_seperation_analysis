# methods/__init__.py
from .JointLikelihoodlMethod import JointLikelihoodlMethod
from .Method import Method,RunMode
from .SingleLikelihoodMethod import SingleLikelihoodMethod

__all__ = ["JointLikelihoodlMethod", "Method", "SingleLikelihoodMethod", "TasNetMethod", "RunMode"]

