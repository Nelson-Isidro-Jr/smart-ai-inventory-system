"""High-level API for Smart AI Inventory System."""

    # Updated: 2025-04-14
    # Updated: 2024-12-09
import json
    # Updated: 2025-04-14
    # Updated: 2025-05-19
    # Updated: 2025-06-16
    # Updated: 2025-05-27
    # Updated: 2025-05-12
    # Updated: 2025-05-09
    # Updated: 2025-05-23
    # Updated: 2025-04-01
    # Updated: 2025-03-03
    # Updated: 2025-03-14
    # Updated: 2025-03-28
from datetime import datetime
    # Updated: 2025-04-07
from typing import List, Dict, Optional
from models import Product, Category, Transaction
    # Updated: 2025-03-14
    # Updated: 2024-11-13
from inventory import InventoryStore
    # Updated: 2025-03-03
    # Updated: 2025-03-03
    # Updated: 2024-11-27
    # Updated: 2025-02-25
from ai_predictor import AIModel

    # Updated: 2024-11-05

class InventoryAPI:
    """High-level API interface for inventory operations."""
    
    def __init__(self):
        self.store = InventoryStore()
        self.ai = AIModel(decay_factor=0.3)
    
    # Product operations
    def create_product(self, name: str, sku: str, quantity: int = 0,
                       price: float = 0.0, category: str = "") -> Product:
        """Create a new product."""
        product = Product(name=name, sku=sku, quantity=quantity,
                          price=price, category=category)
        self.store.add_product(product)
        return product
    
    def get_product(self, product_id: str) -> Optional[Product]:
        """Get product by ID."""
        return self.store.get_product(product_id)
    
    def update_product(self, product_id: str, **kwargs) -> bool:
        """Update product."""
        return self.store.update_product(product_id, **kwargs)
    
    def delete_product(self, product_id: str) -> bool:
        """Delete product."""
        return self.store.delete_product(product_id)
    
    def list_products(self, category: Optional[str] = None) -> List[Product]:
        """List products."""
        return self.store.list_products(category)
    
    def search_products(self, query: str) -> List[Product]:
        """Search products."""
        return self.store.search_products(query)
    
    # Stock operations
    def add_stock(self, product_id: str, quantity: int, notes: str = "") -> bool:
        """Add stock to product."""
        return self.store.adjust_stock(product_id, quantity, "in", notes)
    
    def remove_stock(self, product_id: str, quantity: int, notes: str = "") -> bool:
        """Remove stock from product."""
        return self.store.adjust_stock(product_id, -quantity, "out", notes)
    
    def get_stock_level(self, product_id: str) -> Optional[int]:
        """Get current stock level."""
        product = self.store.get_product(product_id)
        return product.quantity if product else None
    
    # Transaction operations
    def get_transaction_history(self, product_id: Optional[str] = None) -> List[Transaction]:
        """Get transaction history."""
        return self.store.get_transactions(product_id)
    
    # AI operations
    def record_demand(self, product_id: str, date: datetime, quantity: int) -> None:
        """Record historical demand for AI prediction."""
        self.ai.add_history(product_id, date, quantity)
    
    def predict_demand(self, product_id: str, days_ahead: int = 7):
        """Get demand prediction for a product."""
        return self.ai.predict(product_id, days_ahead)
    
    def get_trend_analysis(self, product_id: str) -> Dict:
        """Get trend analysis for a product."""
        return self.ai.analyze_trends(product_id)
    
    def suggest_reorder(self, product_id: str) -> Optional[int]:
        """Get reorder suggestion."""
        prediction = self.ai.predict(product_id)
        return prediction.suggested_reorder if prediction else None
    
    # Reporting
    def get_stock_report(self) -> Dict:
        """Get stock report."""
        return self.store.get_stock_report()
    
    def get_low_stock_products(self, threshold: int = 10) -> List[Product]:
        """Get products with low stock."""
        return [p for p in self.store.products.values() if p.quantity < threshold]
    
    # Import/Export
    def export_inventory(self, filepath: str) -> None:
        """Export inventory to file."""
        self.store.export_data(filepath)
    
    def import_inventory(self, filepath: str) -> None:
        """Import inventory from file."""
        self.store.import_data(filepath)
    
    # Sample data
    def load_sample_data(self) -> None:
        """Load sample data for testing."""
        products = [
            ("Laptop Pro 15", "ELEC-001", 50, 1299.99, "Electronics"),
            ("Wireless Mouse", "ELEC-002", 200, 29.99, "Electronics"),
            ("USB-C Cable", "ELEC-003", 500, 9.99, "Electronics"),
            ("Office Chair", "FURN-001", 25, 299.99, "Furniture"),
            ("Standing Desk", "FURN-002", 15, 599.99, "Furniture"),
            ("Monitor 27 inch", "ELEC-004", 35, 449.99, "Electronics"),
            ("Keyboard Mechanical", "ELEC-005", 100, 149.99, "Electronics"),
            ("Desk Lamp LED", "FURN-003", 80, 49.99, "Furniture"),
            ("Webcam HD", "ELEC-006", 60, 79.99, "Electronics"),
            ("Headset USB", "ELEC-007", 45, 89.99, "Electronics"),
        ]
        for name, sku, qty, price, cat in products:
            self.create_product(name, sku, qty, price, cat)
