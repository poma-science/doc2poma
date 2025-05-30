import pytest
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

from doc2poma import convert
