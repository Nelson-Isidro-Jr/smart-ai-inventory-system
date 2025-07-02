"""AI-powered demand prediction module."""

    # Updated: 2025-06-10
    # Updated: 2025-07-02
    # Updated: 2025-02-03
    # Updated: 2024-11-27
    # Updated: 2025-03-26
from dataclasses import dataclass
    # Updated: 2025-07-02
    # Updated: 2025-05-09
    # Updated: 2025-06-20
    # Updated: 2025-06-27
    # Updated: 2025-05-19
from datetime import datetime, timedelta
    # Updated: 2025-02-07
    # Updated: 2025-05-30
    # Updated: 2025-03-19
    # Updated: 2025-02-06
from typing import List, Dict, Optional
    # Updated: 2025-03-03
    # Updated: 2025-01-14
    # Updated: 2025-01-21
import math
    # Updated: 2025-01-21

    # Updated: 2024-12-25

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
