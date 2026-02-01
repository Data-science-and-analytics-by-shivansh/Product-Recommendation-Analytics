from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import mlflow
from src.config import APRIORI_MIN_SUPPORT, APRIORI_METRIC, APRIORI_MIN_THRESHOLD
from src.utils.logger import logger

def train_apriori(df: pd.DataFrame):
    """
    Expects df with columns: ['InvoiceNo', 'StockCode', 'Description', 'Quantity']
    """
    try:
        with mlflow.start_run(run_name="Apriori_Training"):
            # Prepare basket
            basket = (df.groupby(['InvoiceNo', 'StockCode'])['Quantity']
                      .sum().unstack().reset_index().fillna(0)
                      .set_index('InvoiceNo'))
            basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

            frequent_itemsets = apriori(basket_sets, min_support=APRIORI_MIN_SUPPORT, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric=APRIORI_METRIC, min_threshold=APRIORI_MIN_THRESHOLD)

            rules.to_csv("rules.csv", index=False)

            mlflow.log_param("min_support", APRIORI_MIN_SUPPORT)
            mlflow.log_param("metric", APRIORI_METRIC)
            mlflow.log_metric("num_rules", len(rules))
            mlflow.log_artifact("rules.csv")

            logger.info("Apriori trained", num_rules=len(rules), min_support=APRIORI_MIN_SUPPORT)
            return rules
    except Exception as e:
        logger.exception("Apriori training failed")
        raise
