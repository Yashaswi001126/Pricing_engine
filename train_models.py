from ai.trainer import AIModelTrainer
import pandas as pd

# --------------------------
# Load training data
# --------------------------
df = pd.read_csv("data/options_data.csv")

# --------------------------
# Train and save models
# --------------------------
trainer = AIModelTrainer()
trainer.train_price_model(df)

print("🎯 Training completed successfully")
