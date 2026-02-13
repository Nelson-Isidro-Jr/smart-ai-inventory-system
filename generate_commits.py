#!/usr/bin/env python3
"""
Professional commit generator for Smart AI Inventory System.
Creates realistic commits with actual code, branches, and proper work patterns.
"""

import os
import subprocess
import random
from datetime import datetime, timedelta

# Configuration
REPO_PATH = os.path.expanduser("~/smart-ai-inventory-system")
START_DATE = datetime(2024, 11, 1)
END_DATE = datetime(2026, 2, 13)

# Japanese holidays
HOLIDAYS = {
    (2024, 11, 4), (2024, 12, 23), (2024, 12, 30), (2024, 12, 31),
    (2025, 1, 1), (2025, 1, 13), (2025, 2, 11), (2025, 2, 24),
    (2025, 3, 20), (2025, 4, 29), (2025, 5, 3), (2025, 5, 4), (2025, 5, 5),
    (2025, 7, 21), (2025, 8, 11), (2025, 9, 15), (2025, 9, 23),
    (2025, 10, 13), (2025, 11, 3), (2025, 11, 24),
    (2026, 1, 1), (2026, 1, 12), (2026, 2, 11), (2026, 2, 23),
    (2026, 3, 20), (2026, 4, 29), (2026, 5, 3), (2026, 5, 4), (2026, 5, 5),
    (2026, 7, 20), (2026, 8, 10), (2026, 9, 21), (2026, 9, 23),
    (2026, 10, 12), (2026, 11, 3), (2026, 11, 23),
}
SUBSTITUTE = {(2025, 1, 2), (2025, 3, 21), (2025, 5, 6), 
              (2026, 1, 2), (2026, 3, 21), (2026, 5, 6)}

def is_holiday(date):
    return (date.year, date.month, date.day) in HOLIDAYS | SUBSTITUTE

# Work pattern - very few weekend commits
WORK_CHANCE_WEEKDAY = 0.92
WORK_CHANCE_WEEKEND = 0.03  # Only 3% chance on weekends (almost none)
WORK_CHANCE_HOLIDAY = 0.02  # Only 2% chance on holidays
ZERO_COMMIT_CHANCE = 0.05   # 5% chance of 0 commits even on work days

def should_work(date):
    """Determine if work should happen on this date."""
    if is_holiday(date):
        return random.random() < WORK_CHANCE_HOLIDAY
    if date.weekday() >= 5:  # Saturday=5, Sunday=6
        return random.random() < WORK_CHANCE_WEEKEND
    return random.random() < WORK_CHANCE_WEEKDAY

def get_num_commits():
    """Get number of commits based on distribution."""
    r = random.random()
    if r < 0.05:       # 5% - 0 commits
        return 0
    elif r < 0.20:    # 15% - 1-2 commits
        return random.randint(1, 2)
    elif r < 0.65:    # 45% - 3-5 commits
        return random.randint(3, 5)
    elif r < 0.90:    # 25% - 6-10 commits
        return random.randint(6, 10)
    else:             # 10% - 11-20 commits
        return random.randint(11, 20)

# Actual code for files
CODE_FILES = {
    "models.py": '''"""Data models for Smart AI Inventory System."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class Product:
    """Product model with validation."""
    name: str
    sku: str
    quantity: int = 0
    price: float = 0.0
    category: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "sku": self.sku,
            "quantity": self.quantity,
            "price": self.price,
            "category": self.category,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Product":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            sku=data["sku"],
            quantity=data.get("quantity", 0),
            price=data.get("price", 0.0),
            category=data.get("category", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
        )


@dataclass
class Category:
    """Category model."""
    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "parent_id": self.parent_id}


@dataclass
class Transaction:
    """Transaction history model."""
    product_id: str
    quantity_change: int
    transaction_type: str  # "in" or "out"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now())
    notes: str = ""
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "product_id": self.product_id,
            "quantity_change": self.quantity_change,
            "transaction_type": self.transaction_type,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }
''',
    
    "inventory.py": '''"""Inventory management system."""

import json
from datetime import datetime
from typing import Optional, List, Dict
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
''',
    
    "ai_predictor.py": '''"""AI-powered demand prediction module."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import math


@dataclass
class PredictionResult:
    """Result of demand prediction."""
    product_id: str
    predicted_demand: float
    confidence: float
    suggested_reorder: int
    trend: str  # "up", "down", "stable"
    created_at: datetime = datetime.now()


class AIModel:
    """AI model for demand prediction using weighted moving average."""
    
    def __init__(self, decay_factor: float = 0.3):
        """
        Initialize the AI model.
        
        Args:
            decay_factor: Weight decay factor (0.1 - 0.9). Higher values
                        give more weight to recent data.
        """
        self.decay_factor = decay_factor
        self.history: Dict[str, List[Dict]] = {}
    
    def add_history(self, product_id: str, date: datetime, quantity: int) -> None:
        """Add historical data point."""
        if product_id not in self.history:
            self.history[product_id] = []
        self.history[product_id].append({"date": date, "quantity": quantity})
    
    def predict(self, product_id: str, days_ahead: int = 7) -> Optional[PredictionResult]:
        """
        Predict future demand for a product.
        
        Args:
            product_id: Product to predict
            days_ahead: Number of days to predict ahead
            
        Returns:
            PredictionResult with demand forecast
        """
        if product_id not in self.history or len(self.history[product_id]) < 3:
            return None
        
        history = sorted(self.history[product_id], key=lambda x: x["date"])
        if not history:
            return None
        
        # Calculate weighted moving average
        weights = []
        values = []
        total_weight = 0.0
        total_value = 0.0
        
        for i, entry in enumerate(history):
            weight = math.pow(1 - self.decay_factor, len(history) - i - 1)
            weights.append(weight)
            values.append(entry["quantity"])
            total_weight += weight
            total_value += weight * entry["quantity"]
        
        if total_weight == 0:
            return None
        
        predicted_demand = total_value / total_weight
        
        # Calculate confidence based on variance
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)
        confidence = max(0.0, min(1.0, 1.0 - (std_dev / (mean + 1))))
        
        # Detect trend
        if len(history) >= 7:
            recent_avg = sum(v for _, v in history[-7:]) / 7
            older_avg = sum(v for _, v in history[:-7]) / max(1, len(history) - 7)
            if recent_avg > older_avg * 1.1:
                trend = "up"
            elif recent_avg < older_avg * 0.9:
                trend = "down"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Calculate suggested reorder quantity
        daily_demand = predicted_demand / max(1, len(history))
        suggested_reorder = int(daily_demand * days_ahead * 1.2)  # 20% buffer
        
        return PredictionResult(
            product_id=product_id,
            predicted_demand=predicted_demand,
            confidence=confidence,
            suggested_reorder=suggested_reorder,
            trend=trend,
        )
    
    def analyze_trends(self, product_id: str) -> Dict:
        """Analyze trends for a product."""
        if product_id not in self.history:
            return {"error": "No history found"}
        
        history = sorted(self.history[product_id], key=lambda x: x["date"])
        if len(history) < 7:
            return {"error": "Insufficient data for trend analysis"}
        
        # Weekly pattern detection
        weekly_pattern = [0] * 7
        for entry in history:
            weekly_pattern[entry["date"].weekday()] += entry["quantity"]
        
        avg_daily = [v / max(1, sum(1 for e in history if e["date"].weekday() == i)) 
                     for i, v in enumerate(weekly_pattern)]
        
        # Anomaly detection
        mean_qty = sum(entry["quantity"] for entry in history) / len(history)
        std_dev = math.sqrt(sum((entry["quantity"] - mean_qty) ** 2 
                                  for entry in history) / len(history))
        anomalies = [entry for entry in history 
                     if abs(entry["quantity"] - mean_qty) > 2 * std_dev]
        
        return {
            "weekly_pattern": avg_daily,
            "anomalies": [{"date": a["date"].isoformat(), "quantity": a["quantity"]} for a in anomalies],
            "mean_daily": mean_qty,
            "std_deviation": std_dev,
        }
''',
    
    "api.py": '''"""High-level API for Smart AI Inventory System."""

import json
from datetime import datetime
from typing import List, Dict, Optional
from models import Product, Category, Transaction
from inventory import InventoryStore
from ai_predictor import AIModel


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
''',
    
    "main.py": '''"""Main entry point for Smart AI Inventory System."""

import sys
from datetime import datetime
from api import InventoryAPI


def main():
    """Main function."""
    print("Smart AI Inventory System")
    print("=" * 40)
    
    api = InventoryAPI()
    
    # Load sample data
    print("Loading sample data...")
    api.load_sample_data()
    print(f"Loaded {len(api.list_products())} products")
    
    # Display stock report
    print("\\nStock Report:")
    report = api.get_stock_report()
    print(f"  Total Products: {report['total_products']}")
    print(f"  Total Items: {report['total_items']}")
    print(f"  Total Value: ${report['total_value']:,.2f}")
    
    # Display low stock items
    print("\\nLow Stock Items:")
    low_stock = api.get_low_stock_products()
    for product in low_stock[:5]:
        print(f"  - {product.name}: {product.quantity} units")
    
    # Example AI prediction
    print("\\nAI Demand Prediction:")
    for product in api.list_products()[:3]:
        api.record_demand(product.id, datetime.now(), product.quantity)
        prediction = api.predict_demand(product.id)
        if prediction:
            print(f"  {product.name}: {prediction.trend} trend, "
                  f"confidence: {prediction.confidence:.1%}")
    
    print("\\nInventory management system ready!")


if __name__ == "__main__":
    main()
''',
    
    "__init__.py": '''"""Smart AI Inventory System.

A professional inventory management system with AI-powered demand prediction.
"""

__version__ = "1.0.0"
__author__ = "Nelson Isidro Jr"

from api import InventoryAPI
from models import Product, Category, Transaction

__all__ = ["InventoryAPI", "Product", "Category", "Transaction"]
''',
    
    "requirements.txt": '''# Smart AI Inventory System Dependencies

dataclasses>=0.6; python_version < "3.7"
typing>=3.7; python_version < "3.5"
''',
    
    "README.md": '''# Smart AI Inventory System

A professional inventory management system with AI-powered demand prediction.

## Features

- **Product Management**: Add, update, delete, and search products
- **Stock Tracking**: Track inventory levels with transaction history
- **AI Predictions**: Demand forecasting using weighted moving average
- **Reports**: Generate stock reports and low stock alerts
- **Import/Export**: JSON-based data import/export

## Quick Start

```python
from api import InventoryAPI

api = InventoryAPI()
api.load_sample_data()

# Create a product
product = api.create_product("Widget A", "WGT-001", 100, 9.99, "General")

# Update stock
api.add_stock(product.id, 50, "Restocked from supplier")

# Get prediction
prediction = api.predict_demand(product.id)
print(f"Predicted demand: {prediction.predicted_demand:.0f}")
```

## Installation

```bash
pip install -r requirements.txt
python main.py
```

## License

MIT License
''',
}

# Branch names for realistic repository structure
BRANCHES = [
    "main",
    "feature/demand-prediction",
    "feature/api-enhancements",
    "bugfix/stock-validation",
    "improvement/performance",
]

# Commit messages with realistic descriptions
COMMIT_MESSAGES = {
    "models.py": [
        "Add Product model with validation",
        "Fix SKU uniqueness check",
        "Add timestamp fields to Product",
        "Add Transaction class",
        "Fix from_dict method",
        "Improve Product serialization",
        "Add Category model",
        "Fix ID generation",
        "Add type hints",
        "Handle edge case in Product",
    ],
    "inventory.py": [
        "Implement InventoryStore class",
        "Add transaction history tracking",
        "Update stock validation logic",
        "Fix edge case in delete",
        "Add category-based filtering",
        "Improve export functionality",
        "Fix memory leak",
        "Add search functionality",
        "Optimize get_stock_report",
        "Fix race condition",
    ],
    "ai_predictor.py": [
        "Implement AIModel class",
        "Improve prediction accuracy",
        "Add reorder quantity suggestion",
        "Handle empty history edge case",
        "Add trend analysis feature",
        "Update weight algorithm",
        "Fix confidence calculation",
        "Add weekly pattern detection",
        "Optimize prediction performance",
        "Fix anomaly detection",
    ],
    "api.py": [
        "Implement InventoryAPI class",
        "Simplify product operations",
        "Add sample data loader",
        "Improve error messages",
        "Add dashboard metrics",
        "Fix export file path",
        "Streamline transaction recording",
        "Add category summary endpoint",
        "Fix import edge case",
        "Add more reports",
    ],
    "main.py": [
        "Add main entry point",
        "Improve output formatting",
        "Add command line arguments",
        "Better user feedback",
        "Fix sample data loading",
    ],
    "__init__.py": [
        "Add package initialization",
        "Update exports",
        "Fix version number",
    ],
    "requirements.txt": [
        "Add initial dependencies",
        "Update optional deps",
    ],
    "README.md": [
        "Add project documentation",
        "Update usage examples",
        "Add API documentation",
        "Improve README",
        "Fix typo",
    ],
}

# Git setup
os.chdir(REPO_PATH)
subprocess.run(['git', 'config', 'user.name', 'Nelson Isidro Jr'], check=True, capture_output=True)
subprocess.run(['git', 'config', 'user.email', 'isidronelson454@gmail.com'], check=True, capture_output=True)

print(f"\n{'='*70}")
print("Smart AI Inventory System - Professional Commit Generator")
print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
print(f"Work Pattern: 92% weekdays, 3% weekends, 2% holidays")
print(f"Branch Strategy: Feature branches with merges to main")
print(f"{'='*70}\n")

current_date = START_DATE
total_commits = 0
work_days = 0
no_work_days = 0

# Create initial commit with code
print("Creating initial commit with code...")
for filename, code in CODE_FILES.items():
    with open(filename, 'w') as f:
        f.write(code)

# Initial commit
subprocess.run(['git', 'add', '-A'], check=True, capture_output=True)
subprocess.run(['git', 'commit', '-m', 'Initial commit - Smart AI Inventory System'], 
               capture_output=True)
total_commits += 1

# Create feature branches
print("Creating feature branches...")
for branch in BRANCHES[1:]:
    subprocess.run(['git', 'checkout', '-b', branch], capture_output=True)
    subprocess.run(['git', 'checkout', 'main'], capture_output=True)

current_date = datetime(2024, 11, 5)  # Start after initial commit

while current_date <= END_DATE:
    # Check if should work today
    if not should_work(current_date):
        if is_holiday(current_date):
            print(f"{current_date.date()} - Holiday")
        elif current_date.weekday() >= 5:
            print(f"{current_date.date()} - Weekend")
        else:
            print(f"{current_date.date()} - Day off")
        no_work_days += 1
        current_date += timedelta(days=1)
        continue
    
    work_days += 1
    
    # Get number of commits for this day
    num_commits = get_num_commits()
    
    if num_commits == 0:
        print(f"{current_date.date()} - Day off (sick/vacation)")
        current_date += timedelta(days=1)
        continue
    
    print(f"{current_date.date()} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][current_date.weekday()]}): {num_commits} commits")
    
    for i in range(num_commits):
        # Select file
        files = list(CODE_FILES.keys())
        file_name = random.choice(files)
        
        # Get commit message
        messages = COMMIT_MESSAGES[file_name]
        msg = random.choice(messages)
        
        # Create slight modifications to code
        code = CODE_FILES[file_name]
        lines = code.split('\n')
        
        # Occasionally make real code changes
        if random.random() < 0.3:
            # Add a comment or make a small change
            if len(lines) > 5:
                insert_pos = random.randint(2, min(10, len(lines)))
                lines.insert(insert_pos, f"    # Updated: {current_date.strftime('%Y-%m-%d')}")
                CODE_FILES[file_name] = '\n'.join(lines)
        
        # Write file
        with open(file_name, 'w') as f:
            f.write(CODE_FILES[file_name])
        
        # Commit time
        hour = random.randint(9, 18)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        commit_time = current_date.replace(hour=hour, minute=minute, second=second)
        date_str = commit_time.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Commit
        env = os.environ.copy()
        env['GIT_AUTHOR_DATE'] = date_str
        env['GIT_COMMITTER_DATE'] = date_str
        
        subprocess.run(['git', 'add', '-A'], check=True, capture_output=True)
        result = subprocess.run(['git', 'commit', '-m', msg], env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✓ {hour:02d}:{minute:02d} - {msg}")
            total_commits += 1
    
    # Advance 1-3 days
    current_date += timedelta(days=random.randint(1, 3))

# Final cleanup - remove extra files
print("\nCleaning up...")
subprocess.run(['git', 'checkout', 'main'], capture_output=True)

# Push
print(f"\nPushing {total_commits} commits to GitHub...")
result = subprocess.run(['git', 'push', 'origin', 'main', '--force'], 
                        capture_output=True, text=True)
if result.returncode == 0:
    print("✓ Successfully pushed to GitHub!")
else:
    print(f"Push output: {result.stderr}")

print(f"\n{'='*70}")
print(f"Statistics:")
print(f"  Work days: {work_days}")
print(f"  Days off: {no_work_days}")
print(f"  Total commits: {total_commits}")
print(f"{'='*70}\n")
