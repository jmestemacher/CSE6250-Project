import os
import pickle
import importlib
#os.chdir('../../')
import utils.hosp_preprocess_util
from utils.hosp_preprocess_util import *  
importlib.reload(utils.hosp_preprocess_util)
import utils.hosp_preprocess_util
from utils.hosp_preprocess_util import *

import utils.outlier_removal
from utils.outlier_removal import *  
importlib.reload(utils.outlier_removal)
import utils.outlier_removal
from utils.outlier_removal import *

import utils.uom_conversion
from utils.uom_conversion import *  
importlib.reload(utils.uom_conversion)
import utils.uom_conversion
from utils.uom_conversion import *

# module of preprocessing functions
if not os.path.exists("./data/features"):
    os.makedirs("./data/features")
if not os.path.exists("./data/summary"):
    os.makedirs("./data/summary")

def feature_nonicu(cohort_output,version_path, diag_flag=True,lab_flag=True,proc_flag=True,med_flag=True):
    if diag_flag:
        print("[EXTRACTING DIAGNOSIS DATA]")
        diag = preproc_icd_module("./"+version_path+"/hosp/diagnoses_icd.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', './utils/mappings/ICD9_to_ICD10_mapping.txt', map_code_colname='diagnosis_code')
        diag[['subject_id', 'hadm_id', 'icd_code','root_icd10_convert','root']].to_csv("./data/features/preproc_diag.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

#     if lab_flag:    
#         out = preproc_out("./mimic-iv-1.0/icu/outputevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', 'charttime', dtypes=None, usecols=None)
#         out[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'intime', 'event_time_from_admit']].to_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip', index=False)
    
    if proc_flag:
        print("[EXTRACTING PROCEDURES DATA]")
        proc = preproc_proc_spark("./"+version_path+"/hosp/procedures_icd.csv.gz",'./data/cohort/'+cohort_output+'.csv.gz', 'chartdate', 'base_anchor_year', dtypes=None, usecols=None)
        proc[['subject_id', 'hadm_id', 'icd_code','icd_version', 'chartdate', 'admittime', 'proc_time_from_admit']].to_csv("./data/features/preproc_proc.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    
    if med_flag:
        print("[EXTRACTING MEDICATIONS DATA]")
        med = preproc_meds_spark("./"+version_path+"/hosp/prescriptions.csv.gz", './data/cohort/'+cohort_output+'.csv.gz','./utils/mappings/ndc_product.txt')
        med[['subject_id', 'hadm_id', 'starttime','stoptime','drug','nonproprietaryname', 'start_hours_from_admit', 'stop_hours_from_admit','dose_val_rx']].to_csv('./data/features/preproc_med.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
        
    if lab_flag:
        print("[EXTRACTING LABS DATA]")
        lab = preproc_labs_spark("./"+version_path+"/hosp/labevents.csv.gz", version_path,'./data/cohort/'+cohort_output+'.csv.gz','charttime', 'base_anchor_year', dtypes=None, usecols=None)
        lab = drop_wrong_uom(lab, 0.95)
        lab[['subject_id', 'hadm_id', 'charttime', 'itemid','admittime','lab_time_from_admit','valuenum']].to_csv('./data/features/preproc_labs.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED LABS DATA]")
        
        
        
def preprocess_features_hosp(cohort_output, diag_flag,proc_flag,med_flag,lab_flag,group_diag,group_med,group_proc,clean_labs,impute_labs,thresh,left_thresh):
    #print(thresh)
    if diag_flag:
        print("[PROCESSING DIAGNOSIS DATA]")
        diag = pd.read_csv("./data/features/preproc_diag.csv.gz", compression='gzip',header=0)
        if(group_diag=='Keep both ICD-9 and ICD-10 codes'):
            diag['new_icd_code']=diag['icd_code']
        if(group_diag=='Convert ICD-9 to ICD-10 codes'):
            diag['new_icd_code']=diag['root_icd10_convert']
        if(group_diag=='Convert ICD-9 to ICD-10 and group ICD-10 codes'):
            diag['new_icd_code']=diag['root']

        diag=diag[['subject_id', 'hadm_id', 'new_icd_code']].dropna()
        print("Total number of rows",diag.shape[0])
        diag.to_csv("./data/features/preproc_diag.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if med_flag:
        print("[PROCESSING MEDICATIONS DATA]")
        if group_med:           
            med = pd.read_csv("./data/features/preproc_med.csv.gz", compression='gzip',header=0)
            if group_med:
                med['drug_name']=med['nonproprietaryname']
            else:
                med['drug_name']=med['drug']
            med=med[['subject_id', 'hadm_id', 'starttime','stoptime','drug_name', 'start_hours_from_admit', 'stop_hours_from_admit','dose_val_rx']].dropna()
            print("Total number of rows",med.shape[0])
            med.to_csv('./data/features/preproc_med.csv.gz', compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
    
    
    if proc_flag:
        print("[PROCESSING PROCEDURES DATA]")
        proc = pd.read_csv("./data/features/preproc_proc.csv.gz", compression='gzip',header=0)
        if(group_proc=='ICD-9 and ICD-10'):
            proc=proc[['subject_id', 'hadm_id', 'icd_code', 'chartdate', 'admittime', 'proc_time_from_admit']]
            print("Total number of rows",proc.shape[0])
            proc.dropna().to_csv("./data/features/preproc_proc.csv.gz", compression='gzip', index=False)
        elif(group_proc=='ICD-10'):
            proc=proc.loc[proc.icd_version == 10][['subject_id', 'hadm_id', 'icd_code', 'chartdate', 'admittime', 'proc_time_from_admit']].dropna()
            print("Total number of rows",proc.shape[0])
            proc.to_csv("./data/features/preproc_proc.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
        
        
    if lab_flag:
        
        if clean_labs:   
            print("[PROCESSING LABS DATA]")
            labs = pd.read_csv("./data/features/preproc_labs.csv.gz", compression='gzip',header=0)
            labs = outlier_imputation(labs, 'itemid', 'valuenum', thresh,left_thresh,impute_labs)
            

#             for i in [51249, 51282]:
#                 try:
#                     maj = labs.loc[labs.itemid == i].valueuom.value_counts().index[0]
#                     labs = labs.loc[~((labs.itemid == i) & (labs.valueuom == maj))]
#                 except IndexError:
#                     print(f"{idx} not found")
            print("Total number of rows",labs.shape[0])
#             del labs['valueuom']
            labs.to_csv("./data/features/preproc_labs.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED LABS DATA]")
        
def generate_summary_hosp(diag_flag,proc_flag,med_flag,lab_flag):
    print("[GENERATING FEATURE SUMMARY]")
    if diag_flag:
        diag = pd.read_csv("./data/features/preproc_diag.csv.gz", compression='gzip',header=0)
        print(diag.columns.values.tolist())
        freq=diag.groupby(['hadm_id','new_icd_code']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['new_icd_code'])['mean_frequency'].mean().reset_index()
        total=diag.groupby('new_icd_code').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='new_icd_code',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/diag_summary.csv',index=False)
        summary['new_icd_code'].to_csv('./data/summary/diag_features.csv',index=False)


    if med_flag:
        med = pd.read_csv("./data/features/preproc_med.csv.gz", compression='gzip',header=0)
        freq=med.groupby(['hadm_id','drug_name']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['drug_name'])['mean_frequency'].mean().reset_index()
        
        missing=med[med['dose_val_rx']==0].groupby('drug_name').size().reset_index(name="missing_count")
        total=med.groupby('drug_name').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='drug_name',how='right')
        summary=pd.merge(freq,summary,on='drug_name',how='right')
        summary['missing%']=100*(summary['missing_count']/summary['total_count'])
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/med_summary.csv',index=False)
        summary['drug_name'].to_csv('./data/summary/med_features.csv',index=False)

    
    
    if proc_flag:
        proc = pd.read_csv("./data/features/preproc_proc.csv.gz", compression='gzip',header=0)
        freq=proc.groupby(['hadm_id','icd_code']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['icd_code'])['mean_frequency'].mean().reset_index()
        total=proc.groupby('icd_code').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='icd_code',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/proc_summary.csv',index=False)
        summary['icd_code'].to_csv('./data/summary/proc_features.csv',index=False)

        
        
    if lab_flag:
        chunksize = 10000000
        labs=pd.DataFrame()
        for chunk in tqdm(pd.read_csv("./data/features/preproc_labs.csv.gz", compression='gzip',header=0, index_col=None,chunksize=chunksize)):
            if labs.empty:
                labs=chunk
            else:
                labs=labs.append(chunk, ignore_index=True)
        freq=labs.groupby(['hadm_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        
        missing=labs[labs['valuenum']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=labs.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,summary,on='itemid',how='right')
        summary['missing%']=100*(summary['missing_count']/summary['total_count'])
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/labs_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/labs_features.csv',index=False)

    print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean

# Initialize Spark session
spark = SparkSession.builder \
    .appName("GenerateSummaryHosp") \
    .getOrCreate()

def generate_summary_hosp_spark(diag_flag, proc_flag, med_flag, lab_flag):
    print("[GENERATING FEATURE SUMMARY]")
    spark = SparkSession.builder.appName("GenerateSummaryHosp").getOrCreate()
    if diag_flag:
        diag = spark.read.csv("./data/features/preproc_diag.csv.gz", header=True, inferSchema=True)
        freq = diag.groupBy('hadm_id', 'new_icd_code').count().groupBy('new_icd_code').agg(mean('count').alias('mean_frequency'))
        total = diag.groupBy('new_icd_code').count().alias('total_count')
        summary = freq.join(total, 'new_icd_code', 'right').fillna(0)
        summary.toPandas().to_csv('./data/summary/diag_summary.csv', index=False)
        summary.select('new_icd_code').toPandas().to_csv('./data/summary/diag_features.csv', index=False)

    if med_flag:
        med = spark.read.csv("./data/features/preproc_med.csv.gz", header=True, inferSchema=True)
        freq = med.groupBy('hadm_id', 'drug_name').count().groupBy('drug_name').agg(mean('count').alias('mean_frequency'))
        missing = med.filter(col('dose_val_rx') == 0).groupBy('drug_name').count().alias('missing_count')
        total = med.groupBy('drug_name').count().alias('total_count')
        summary = freq.join(missing, 'drug_name', 'right').join(total, 'drug_name', 'right')
        summary = summary.withColumn('missing%', 100 * (col('missing_count') / col('total_count'))).fillna(0)
        summary.toPandas().to_csv('./data/summary/med_summary.csv', index=False)
        summary.select('drug_name').toPandas().to_csv('./data/summary/med_features.csv', index=False)

    if proc_flag:
        proc = spark.read.csv("./data/features/preproc_proc.csv.gz", header=True, inferSchema=True)
        freq = proc.groupBy('hadm_id', 'icd_code').count().groupBy('icd_code').agg(mean('count').alias('mean_frequency'))
        total = proc.groupBy('icd_code').count().alias('total_count')
        summary = freq.join(total, 'icd_code', 'right').fillna(0)
        summary.toPandas().to_csv('./data/summary/proc_summary.csv', index=False)
        summary.select('icd_code').toPandas().to_csv('./data/summary/proc_features.csv', index=False)

    if lab_flag:
        labs = spark.read.csv("./data/features/preproc_labs.csv.gz", header=True, inferSchema=True)
        freq = labs.groupBy('hadm_id', 'itemid').count().groupBy('itemid').agg(mean('count').alias('mean_frequency'))
        missing = labs.filter(col('valuenum') == 0).groupBy('itemid').count().alias('missing_count')
        total = labs.groupBy('itemid').count().alias('total_count')
        summary = freq.join(missing, 'itemid', 'right').join(total, 'itemid', 'right')
        summary = summary.withColumn('missing%', 100 * (col('missing_count') / col('total_count'))).fillna(0)
        summary.toPandas().to_csv('./data/summary/labs_summary.csv', index=False)
        summary.select('itemid').toPandas().to_csv('./data/summary/labs_features.csv', index=False)

    print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FeaturesSelectionHosp") \
    .getOrCreate()

def features_selection_hosp_spark(cohort_output, diag_flag, proc_flag, med_flag, lab_flag, group_diag, group_med, group_proc, clean_labs):
    if diag_flag:
        if group_diag:
            print("[FEATURE SELECTION DIAGNOSIS DATA]")
            diag = spark.read.csv("./data/features/preproc_diag.csv.gz", header=True, inferSchema=True)
            features = spark.read.csv("./data/summary/diag_features.csv", header=True, inferSchema=True)
            diag = diag.filter(col('new_icd_code').isin([row['new_icd_code'] for row in features.collect()]))
            print("Total number of rows", diag.count())
            diag.toPandas().to_csv("./data/features/preproc_diag.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

    if med_flag:
        if group_med:
            print("[FEATURE SELECTION MEDICATIONS DATA]")
            med = spark.read.csv("./data/features/preproc_med.csv.gz", header=True, inferSchema=True)
            features = spark.read.csv("./data/summary/med_features.csv", header=True, inferSchema=True)
            med = med.filter(col('drug_name').isin([row['drug_name'] for row in features.collect()]))
            print("Total number of rows", med.count())
            med.toPandas().to_csv('./data/features/preproc_med.csv.gz', compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

    if proc_flag:
        if group_proc:
            print("[FEATURE SELECTION PROCEDURES DATA]")
            proc = spark.read.csv("./data/features/preproc_proc.csv.gz", header=True, inferSchema=True)
            features = spark.read.csv("./data/summary/proc_features.csv", header=True, inferSchema=True)
            proc = proc.filter(col('icd_code').isin([row['icd_code'] for row in features.collect()]))
            print("Total number of rows", proc.count())
            proc.toPandas().to_csv("./data/features/preproc_proc.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")

    if lab_flag:
        if clean_labs:
            print("[FEATURE SELECTION LABS DATA]")
            labs = spark.read.csv("./data/features/preproc_labs.csv.gz", header=True, inferSchema=True)
            features = spark.read.csv("./data/summary/labs_features.csv", header=True, inferSchema=True)
            labs = labs.filter(col('itemid').isin([row['itemid'] for row in features.collect()]))
            print("Total number of rows", labs.count())
            labs.toPandas().to_csv("./data/features/preproc_labs.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED LABS DATA]")

def features_selection_hosp(cohort_output, diag_flag,proc_flag,med_flag,lab_flag,group_diag,group_med,group_proc,clean_labs):
    if diag_flag:
        if group_diag:
            print("[FEATURE SELECTION DIAGNOSIS DATA]")
            diag = pd.read_csv("./data/features/preproc_diag.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/diag_features.csv",header=0)
            diag=diag[diag['new_icd_code'].isin(features['new_icd_code'].unique())]
        
            print("Total number of rows",diag.shape[0])
            diag.to_csv("./data/features/preproc_diag.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if med_flag:       
        if group_med:   
            print("[FEATURE SELECTION MEDICATIONS DATA]")
            med = pd.read_csv("./data/features/preproc_med.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/med_features.csv",header=0)
            med=med[med['drug_name'].isin(features['drug_name'].unique())]
            print("Total number of rows",med.shape[0])
            med.to_csv('./data/features/preproc_med.csv.gz', compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
    
    
    if proc_flag:
        if group_proc:
            print("[FEATURE SELECTION PROCEDURES DATA]")
            proc = pd.read_csv("./data/features/preproc_proc.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/proc_features.csv",header=0)
            proc=proc[proc['icd_code'].isin(features['icd_code'].unique())]
            print("Total number of rows",proc.shape[0])
            proc.to_csv("./data/features/preproc_proc.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
        
        
    if lab_flag:
        if clean_labs:            
            print("[FEATURE SELECTION LABS DATA]")
            chunksize = 10000000
            labs=pd.DataFrame()
            for chunk in tqdm(pd.read_csv("./data/features/preproc_labs.csv.gz", compression='gzip',header=0, index_col=None,chunksize=chunksize)):
                if labs.empty:
                    labs=chunk
                else:
                    labs=labs.append(chunk, ignore_index=True)
            features=pd.read_csv("./data/summary/labs_features.csv",header=0)
            labs=labs[labs['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",labs.shape[0])
            labs.to_csv("./data/features/preproc_labs.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED LABS DATA]")
