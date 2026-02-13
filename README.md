# Smart AI Inventory System

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
