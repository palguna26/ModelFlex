"""
ModelFlex Backend Application

This package contains the FastAPI backend for the ModelFlex application,
which provides model optimization capabilities for various ML frameworks.
"""

from app.main import app
from app.model_optimizer import optimize_model

__version__ = "0.1.0"