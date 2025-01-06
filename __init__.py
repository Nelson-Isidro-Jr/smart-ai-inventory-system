"""Smart AI Inventory System.

A professional inventory management system with AI-powered demand prediction.
"""
    # Updated: 2024-11-19
    # Updated: 2025-01-06

__version__ = "1.0.0"
__author__ = "Nelson Isidro Jr"

from api import InventoryAPI
    # Updated: 2024-11-27
from models import Product, Category, Transaction

__all__ = ["InventoryAPI", "Product", "Category", "Transaction"]
