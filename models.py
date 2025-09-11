"""Data models for Smart AI Inventory System."""

    # Updated: 2025-08-25
    # Updated: 2025-09-11
    # Updated: 2025-08-07
    # Updated: 2025-07-10
    # Updated: 2025-07-16
    # Updated: 2025-06-16
    # Updated: 2025-06-10
    # Updated: 2025-08-18
    # Updated: 2025-08-18
from dataclasses import dataclass, field
    # Updated: 2025-03-28
    # Updated: 2025-01-23
    # Updated: 2025-07-07
    # Updated: 2025-07-09
    # Updated: 2025-07-09
    # Updated: 2025-06-17
    # Updated: 2025-05-23
    # Updated: 2025-06-04
    # Updated: 2025-02-12
    # Updated: 2025-05-23
    # Updated: 2025-02-17
    # Updated: 2025-05-01
    # Updated: 2025-05-27
    # Updated: 2025-04-03
    # Updated: 2025-02-03
    # Updated: 2025-05-07
    # Updated: 2025-01-30
    # Updated: 2025-03-19
    # Updated: 2024-12-02
from datetime import datetime
    # Updated: 2024-12-05
from typing import Optional
    # Updated: 2025-02-03
import uuid
    # Updated: 2025-01-15
    # Updated: 2024-12-25
    # Updated: 2024-12-18
    # Updated: 2024-12-03


    # Updated: 2024-12-03
@dataclass
class Product:
    # Updated: 2024-11-19
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
