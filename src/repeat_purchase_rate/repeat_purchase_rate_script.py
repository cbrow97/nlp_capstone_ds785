import pyspark.sql.functions as f
from pyspark.sql.window import Window, WindowSpec
from pyspark.sql.types import DateType
from pyspark.sql import DataFrame as SparkDataFrame
from functools import reduce, partial
from datetime import datetime, timedelta
from actionable_intelligence_reporting.common import databricks_utils
from actionable_intelligence_reporting.standardized_fact_tables import (
    read_write_update as rwu,
)
from typing import Tuple


def get_min_and_max_txn_date(
    df: SparkDataFrame,
) -> Tuple[datetime, datetime]:
    """
    Returns the minimum and maximum transaction date found in the
    daily_transaction_fact table.
    """
    return (
        df.select(f.min("week_end")).collect()[0][0],
        df.select(f.max("week_end")).collect()[0][0],
    )


def get_max_week_end_for_repeat_purchase_counts_df(
    day_thres: int, max_txn_date: datetime
) -> datetime:
    return max_txn_date - timedelta(days=day_thres)


def generate_repeat_purchase_counts_df(
    daily_customer_summary_fact_df: SparkDataFrame,
    day_thres: int,
    max_txn_date: datetime,
) -> SparkDataFrame:
    max_week_end = get_max_week_end_for_repeat_purchase_counts_df(
        day_thres, max_txn_date
    )
    return (
        daily_customer_summary_fact_df.groupBy("week_end", "most_freq_store_id")
        .agg(
            f.countDistinct(
                f.when(
                    (f.col(f"repeat_purchase_{day_thres}_day") > 0), f.col("guest_id")
                )
            ).alias(f"current_week_repeat_purchase_{day_thres}_day_customer_count"),
            f.countDistinct(
                f.when(f.col(f"total_txns_{day_thres}_days") > 0, f.col("guest_id"))
            ).alias(f"current_week_purchased_in_last_{day_thres}_day_customer_count"),
        )
        .withColumn(
            f"current_week_repeat_purchase_{day_thres}_day_customer_count",
            f.when(
                f.col("week_end") <= max_week_end,
                f.col(f"current_week_repeat_purchase_{day_thres}_day_customer_count"),
            ).otherwise(f.lit(None)),
        )
        .withColumn(
            f"current_week_purchased_in_last_{day_thres}_day_customer_count",
            f.when(
                f.col("week_end") <= max_week_end,
                f.col(f"current_week_purchased_in_last_{day_thres}_day_customer_count"),
            ).otherwise(f.lit(None)),
        )
        .withColumn(
            f"current_week_repeat_purchase_rate_{day_thres}_day",
            f.col(f"current_week_repeat_purchase_{day_thres}_day_customer_count")
            / f.col(f"current_week_purchased_in_last_{day_thres}_day_customer_count"),
        )
    )


def generate_repeat_purchase_df(
    daily_customer_summary_fact_df: SparkDataFrame, max_txn_date: datetime
) -> SparkDataFrame:
    _REPEAT_PURCHASE_LOOKBACK_DAYS = (7, 28)

    rpr_dfs = [
        generate_repeat_purchase_counts_df(
            daily_customer_summary_fact_df, repeat_purchase_lookback_day, max_txn_date
        ).cache()
        for repeat_purchase_lookback_day in _REPEAT_PURCHASE_LOOKBACK_DAYS
    ]
    return reduce(
        partial(databricks_utils.join_all, join_cond=["week_end", "most_freq_store_id"]),
        rpr_dfs,
    )


def generate_weekly_customer_status_fact_df(
    spark, env: str, customer_code: str
) -> SparkDataFrame:
    daily_customer_summary_fact_df = spark.table(
        f"{env}_{customer_code}.daily_customer_summary_fact"
    ).join(
        spark.table("brightloom_kpi.dim_date").selectExpr(
            "dayNumber as as_of_date", "week_end"
        ),
        on="as_of_date",
    )
    
    _, max_txn_date = get_min_and_max_txn_date(daily_customer_summary_fact_df)

    repeat_purchase_counts_df = generate_repeat_purchase_df(
        daily_customer_summary_fact_df, max_txn_date
    )
    
    return repeat_purchase_counts_df