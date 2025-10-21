# Smart AI Inventory System

    # Updated: 2025-09-08
    # Updated: 2025-06-13
    # Updated: 2025-05-16
    # Updated: 2025-08-28
    # Updated: 2025-06-13
    # Updated: 2025-10-10
    # Updated: 2025-07-29
    # Updated: 2024-12-18
    # Updated: 2025-10-21
    # Updated: 2025-10-20
    # Updated: 2025-10-06
    # Updated: 2025-08-21
    # Updated: 2025-09-24
A professional inventory management system with AI-powered demand prediction.
    # Updated: 2025-08-01
    # Updated: 2025-08-07
    # Updated: 2025-05-23
    # Updated: 2025-08-07
    # Updated: 2025-07-02
    # Updated: 2025-02-06
    # Updated: 2025-07-09
    # Updated: 2025-05-19
    # Updated: 2024-12-03
    # Updated: 2025-01-16
    # Updated: 2025-01-21
    # Updated: 2024-11-06
    # Updated: 2025-02-12
    # Updated: 2025-02-17
    # Updated: 2025-02-07
    # Updated: 2024-11-25
    # Updated: 2024-12-25
    # Updated: 2024-11-27
    # Updated: 2024-12-03
    # Updated: 2024-11-22
    # Updated: 2024-12-02
    # Updated: 2024-11-11

## Features
    # Updated: 2024-11-25

- **Product Management**: Add, update, delete, and search products
    # Updated: 2024-11-22
    # Updated: 2024-11-07
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
