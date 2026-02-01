# Run with: python -m scripts.train_pipeline
import pandas as pd
from src.utils.logger import logger
from src.models.apriori import train_apriori
from src.models.collaborative import train_svd
from src.config import DATA_PATH

def main():
    try:
        logger.info("Starting full training pipeline")
        df = pd.read_csv(DATA_PATH, encoding='latin1')  # adjust encoding if needed

        # Apriori needs transaction format
        train_apriori(df)

        # For SVD: create implicit ratings (e.g. 1 for every purchase)
        df_ratings = df[['CustomerID', 'StockCode']].copy()
        df_ratings = df_ratings.dropna()
        df_ratings['rating'] = 1
        df_ratings.columns = ['user_id', 'item_id', 'rating']

        train_svd(df_ratings)

        logger.info("Training pipeline completed successfully")
    except Exception as e:
        logger.exception("Pipeline failed")
        raise

if __name__ == "__main__":
    main()
