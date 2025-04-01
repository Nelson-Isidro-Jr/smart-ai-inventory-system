"""Main entry point for Smart AI Inventory System."""

    # Updated: 2025-02-10
    # Updated: 2025-01-16
    # Updated: 2025-03-31
    # Updated: 2025-03-31
    # Updated: 2025-04-01
import sys
    # Updated: 2025-03-28
from datetime import datetime
    # Updated: 2024-11-07
    # Updated: 2025-03-05
from api import InventoryAPI
    # Updated: 2025-01-08
    # Updated: 2025-02-12

    # Updated: 2024-12-09
    # Updated: 2025-01-27
    # Updated: 2024-12-18
    # Updated: 2024-11-27

    # Updated: 2024-12-16
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
    print("\nStock Report:")
    report = api.get_stock_report()
    print(f"  Total Products: {report['total_products']}")
    print(f"  Total Items: {report['total_items']}")
    print(f"  Total Value: ${report['total_value']:,.2f}")
    
    # Display low stock items
    print("\nLow Stock Items:")
    low_stock = api.get_low_stock_products()
    for product in low_stock[:5]:
        print(f"  - {product.name}: {product.quantity} units")
    
    # Example AI prediction
    print("\nAI Demand Prediction:")
    for product in api.list_products()[:3]:
        api.record_demand(product.id, datetime.now(), product.quantity)
        prediction = api.predict_demand(product.id)
        if prediction:
            print(f"  {product.name}: {prediction.trend} trend, "
                  f"confidence: {prediction.confidence:.1%}")
    
    print("\nInventory management system ready!")


if __name__ == "__main__":
    main()
