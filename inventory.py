"""Inventory management system."""

import json
    # Updated: 2024-11-06
from datetime import datetime
from typing import Optional, List, Dict
    # Updated: 2024-11-07
    # Updated: 2024-11-11
from models import Product, Category, Transaction


class InventoryStore:
    """Store for managing inventory."""
    
    def __init__(self):
        self.products: Dict[str, Product] = {}
        self.categories: Dict[str, Category] = {}
        self.transactions: List[Transaction] = []
    
    def add_product(self, product: Product) -> bool:
        """Add a new product."""
        if product.sku in [p.sku for p in self.products.values()]:
            return False
        self.products[product.id] = product
        return True
    
    def get_product(self, product_id: str) -> Optional[Product]:
        """Get product by ID."""
        return self.products.get(product_id)
    
    def update_product(self, product_id: str, **kwargs) -> bool:
        """Update product."""
        if product_id not in self.products:
            return False
        product = self.products[product_id]
        for key, value in kwargs.items():
            if hasattr(product, key):
                setattr(product, key, value)
        product.updated_at = datetime.now()
        return True
    
    def delete_product(self, product_id: str) -> bool:
        """Delete product."""
        if product_id not in self.products:
            return False
        del self.products[product_id]
        return True
    
    def list_products(self, category: Optional[str] = None) -> List[Product]:
        """List products, optionally filtered by category."""
        products = list(self.products.values())
        if category:
            products = [p for p in products if p.category == category]
        return products
    
    def add_transaction(self, transaction: Transaction) -> None:
        """Record a transaction."""
        self.transactions.append(transaction)
    
    def get_transactions(self, product_id: Optional[str] = None) -> List[Transaction]:
        """Get transactions, optionally filtered by product."""
        if product_id:
            return [t for t in self.transactions if t.product_id == product_id]
        return self.transactions
    
    def adjust_stock(self, product_id: str, quantity_change: int, 
                     transaction_type: str, notes: str = "") -> bool:
        """Adjust stock level."""
        if product_id not in self.products:
            return False
        product = self.products[product_id]
        new_quantity = product.quantity + quantity_change
        if new_quantity < 0:
            return False
        product.quantity = new_quantity
        product.updated_at = datetime.now()
        
        transaction = Transaction(
            product_id=product_id,
            quantity_change=quantity_change,
            transaction_type=transaction_type,
            notes=notes,
        )
        self.add_transaction(transaction)
        return True
    
    def search_products(self, query: str) -> List[Product]:
        """Search products by name or SKU."""
        query = query.lower()
        return [p for p in self.products.values() 
                if query in p.name.lower() or query in p.sku.lower()]
    
    def get_stock_report(self) -> Dict:
        """Generate stock report."""
        total_value = sum(p.quantity * p.price for p in self.products.values())
        low_stock = [p for p in self.products.values() if p.quantity < 10]
        return {
            "total_products": len(self.products),
            "total_items": sum(p.quantity for p in self.products.values()),
            "total_value": total_value,
            "low_stock_count": len(low_stock),
            "low_stock_items": [{"name": p.name, "quantity": p.quantity} for p in low_stock],
        }
    
    def export_data(self, filepath: str) -> None:
        """Export inventory to JSON file."""
        data = {
            "products": [p.to_dict() for p in self.products.values()],
            "categories": [c.to_dict() for c in self.categories.values()],
            "exported_at": datetime.now().isoformat(),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_data(self, filepath: str) -> None:
        """Import inventory from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        for p_data in data.get("products", []):
            self.add_product(Product.from_dict(p_data))
