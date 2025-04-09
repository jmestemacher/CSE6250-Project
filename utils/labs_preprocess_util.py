import os
from glob import glob
from uuid import uuid1
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
from typing import Union, List, Tuple
import pandas as pd
import numpy as np
if not os.path.exists("./data/temp"):
    os.makedirs("./data/temp")


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when, datediff
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from collections import defaultdict

def hadm_imputer_1(
    charttime: pd._libs.tslibs.timestamps.Timestamp,
    hadm_old: Union[str, float],
    hadm_ids_w_timestamps: List[
        Tuple[
            str,
            pd._libs.tslibs.timestamps.Timestamp,
            pd._libs.tslibs.timestamps.Timestamp,
        ]
    ],
) -> Tuple[str, pd._libs.tslibs.timestamps.Timestamp]:
    # if old hadm exists use that
    if not np.isnan(hadm_old):
        #print("old")
        hadm_old = int(hadm_old)
        admtime, dischtime = [
            [adm_time, disch_time]
            for h_id, adm_time, disch_time in hadm_ids_w_timestamps
            if h_id == hadm_old
        ][0]
        #print("got old")
        return (
            hadm_old,
            admtime.strftime("%Y-%m-%d %H:%M:%S"),
            dischtime.strftime("%Y-%m-%d %H:%M:%S"),
        )
    # get the difference between this lab event charttime and all admit times for this subject_id
    hadm_ids_w_timestamps = [
        [
            hadm_id,
            admittime.strftime("%Y-%m-%d %H:%M:%S"),
            dischtime.strftime("%Y-%m-%d %H:%M:%S"),
            charttime.normalize() - admittime.normalize(),
            charttime.normalize() - dischtime.normalize(),
        ]
        for hadm_id, admittime, dischtime in hadm_ids_w_timestamps
    ]
    # the lab charttime must be in between admit time and discharge time
    hadm_ids_w_timestamps = [
        x for x in hadm_ids_w_timestamps if x[3].days >= 0 and x[4].days <= 0
    ]
    # there should be exactly one hadm_id that satisfies this criteria
    # if multiple, select the hadm id with admittime closest to the lab event charttime
    hadm_ids_w_timestamps = sorted(hadm_ids_w_timestamps, key=lambda x: x[3])
    if not hadm_ids_w_timestamps:
        return None, None, None
    return_data = hadm_ids_w_timestamps[0][:3]
    return return_data


def impute_missing_hadm_ids_1(
    lab_table: pd.DataFrame, subject_hadm_admittime_tracker: defaultdict
) -> pd.DataFrame:
    list_rows_lab = []
    all_lab_cols = lab_table.columns
    for row in lab_table.itertuples():
        existing_data = {
            col_name: row.__getattribute__(col_name) for col_name in all_lab_cols
        }
        new_hadm_id, new_admittime, new_dischtime = hadm_imputer_1(
            row.charttime,
            row.hadm_id,
            subject_hadm_admittime_tracker.get(
                row.subject_id, []
            ),  # using get as defaultdict will create key if does not exist
        )
        existing_data["hadm_id_new"] = new_hadm_id
        existing_data["admittime"] = new_admittime
        existing_data["dischtime"] = new_dischtime
        list_rows_lab.append(existing_data)
    tab_name = str(uuid1())
    pd.DataFrame(list_rows_lab).to_csv(f"{tab_name}.csv")


def impute_hadm_ids_1(
    lab_table: Union[str, pd.DataFrame], admission_table: Union[str, pd.DataFrame]
) -> pd.DataFrame:
    if isinstance(lab_table, str):
        lab_table = pd.read_csv(lab_table)
    if isinstance(admission_table, str):
        admission_table = pd.read_csv(admission_table)
    lab_table["charttime"] = pd.to_datetime(lab_table.charttime)
    admission_table["admittime"] = pd.to_datetime(admission_table.admittime)
    admission_table["dischtime"] = pd.to_datetime(admission_table.dischtime)
    # get a dictionary like this->
    """ {
        "sub_id_1": [["hadm_1", "admittime1", "dischtime1"], ["hadm_2", "admittime2", "dischtime2"]],
        "sub_id_2": [["hadm_1", "admittime1", "dischtime1"], ["hadm_2", "admittime2", "dischtime2"]],
        ...
    """
    subject_hadm_admittime_tracker = defaultdict(list)
    for row in admission_table.itertuples():
        subject_hadm_admittime_tracker[row.subject_id].append(
            [row.hadm_id, row.admittime, row.dischtime]
        )
    lab_size = lab_table.shape[0]
    chunks = 100
    tab_size = lab_size // chunks
    lab_table_chunks = []
    for i in range(chunks):
        st, en = i * tab_size, (i + 1) * tab_size
        lab_table_chunks.append(lab_table[st:en])
    if lab_size - chunks * tab_size > 0:
        lab_table_chunks.append(lab_table[chunks * tab_size :])
    # we dont need the original lab table as it is chunkified, hope is to save memory as lab table is huge
    del lab_table
    impute_missing_hadm_ids_w_lookup = partial(
        impute_missing_hadm_ids_1,
        subject_hadm_admittime_tracker=subject_hadm_admittime_tracker,
    )
    #print(impute_missing_hadm_ids_w_lookup)
    #print(len(lab_table_chunks))
    with Pool(8) as p:
        p.map(impute_missing_hadm_ids_w_lookup, lab_table_chunks)
    all_csvs = glob("*.csv")
    lab_tab = pd.DataFrame()
    for csv in all_csvs:
        lab_tab = pd.concat([lab_tab, pd.read_csv(csv)])
        os.remove(csv)
    return lab_tab


from pyspark.sql import functions as F
from pyspark.sql.window import Window

def hadm_imputer_spark(labs_na, admission_table):
    """
    Impute missing hadm_id values for lab events using admission data.

    Parameters:
    labs_na: PySpark DataFrame containing lab events with missing hadm_id.
    admission_table: PySpark DataFrame containing admission data with hadm_id, admittime, and dischtime.

    Returns:
    PySpark DataFrame with imputed hadm_id, admittime, and dischtime.
    """
    
    #print(admission_table.columns)
    # Join labs_na with admission_table on subject_id
    labs_na = labs_na.drop('hadm_id')
    labs_na = labs_na.join(admission_table, on='subject_id', how='left')
    #print(labs_na.columns)

    # Calculate the difference between charttime and admittime/dischtime
    labs_na = labs_na.withColumn('admit_diff', F.datediff(F.col('charttime'), F.col('admittime')))
    labs_na = labs_na.withColumn('disch_diff', F.datediff(F.col('charttime'), F.col('dischtime')))

    # Filter rows where charttime is between admittime and dischtime
    labs_na = labs_na.filter((F.col('admit_diff') >= 0) & (F.col('disch_diff') <= 0))

    # Define a window specification to get the closest admittime
    window_spec = Window.partitionBy('subject_id', 'charttime').orderBy(F.col('admit_diff').asc())

    # Assign row numbers based on the closest admittime
    labs_na = labs_na.withColumn('row_number', F.row_number().over(window_spec))

    # Filter to keep only the closest match (row_number = 1)
    labs_na = labs_na.filter(F.col('row_number') == 1).drop('row_number', 'admit_diff', 'disch_diff')
    
    # Rename columns for consistency
    labs_na = labs_na.withColumnRenamed('hadm_id', 'hadm_id_new') \
                     .withColumnRenamed('admittime', 'admittime_new') \
                     .withColumnRenamed('dischtime', 'dischtime_new')

    return labs_na

def impute_missing_hadm_ids_spark(labs_na, admission_table):
    """
    Impute missing hadm_id values for lab events using admission data.

    Parameters:
    labs_na: PySpark DataFrame containing lab events with missing hadm_id.
    admission_table: PySpark DataFrame containing admission data with hadm_id, admittime, and dischtime.

    Returns:
    PySpark DataFrame with imputed hadm_id values.
    """
    # Call hadm_imputer_spark to process missing hadm_id rows
    labs_na = hadm_imputer_spark(labs_na, admission_table)
    # Select only the required columns and rename hadm_id_new to hadm_id
    labs_na = labs_na.select(
        'subject_id',
        'hadm_id_new',
        'itemid',
        'charttime',
        'valuenum',
        'valueuom'
    ).withColumnRenamed('hadm_id_new', 'hadm_id')
    #print(labs_na.columns)

    return labs_na


def impute_hadm_ids_spark(lab_table_path, admission_table_path):
    """
    Impute missing hadm_id values in the lab table using admission data.
    
    Parameters:
    lab_table_path: Path to the lab events CSV file.
    admission_table_path: Path to the admission table CSV file.
    
    Returns:
    PySpark DataFrame with imputed hadm_id values.
    """
    # Initialize Spark session
    spark = SparkSession.builder \
    .appName("ImputeMissingHadmIds") \
    .getOrCreate()
    # Read lab_table and admission_table
    #lab_table = spark.read.csv(lab_table_path, header=True, inferSchema=True)
    #admission_table = spark.read.csv(admission_table_path, header=True, inferSchema=True)
    lab_table = lab_table_path
    admission_table = admission_table_path

    # Convert charttime, admittime, and dischtime to datetime
    lab_table = lab_table.withColumn('charttime', col('charttime').cast('timestamp'))
    admission_table = admission_table.withColumn('admittime', col('admittime').cast('timestamp'))
    admission_table = admission_table.withColumn('dischtime', col('dischtime').cast('timestamp'))

    # Filter rows with missing hadm_id
    labs_na = lab_table.filter(col('hadm_id').isNull())
    labs = lab_table.filter(col('hadm_id').isNotNull())
    print(labs.columns)

    # Impute missing hadm_id values
    labs_na = impute_missing_hadm_ids_spark(labs_na, admission_table)
    print(labs_na.columns)
    # Select only the common columns from labs_na before union
    #labs_na = labs_na.select('subject_id', 'hadm_id_new', 'itemid', 'charttime', 'valuenum', 'valueuom') \
    #    .withColumnRenamed('hadm_id_new', 'hadm_id') # Rename hadm_id_new to hadm_id to match the labs DataFrame


    # Union the imputed labs_na with the original labs
    lab_table = labs.union(labs_na)

    return lab_table