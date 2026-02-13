"""Smart AI Inventory System.

A professional inventory management system with AI-powered demand prediction.
"""

__version__ = "1.0.0"
__author__ = "Nelson Isidro Jr"

from api import InventoryAPI
from models import Product, Category, Transaction

__all__ = ["InventoryAPI", "Product", "Category", "Transaction"]
