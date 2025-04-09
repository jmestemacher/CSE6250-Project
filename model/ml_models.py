import pandas as pd
import numpy as np
import pickle
import torch
import random
import os
import importlib
import sys
import numpy as np
import evaluation
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')

importlib.reload(evaluation)
import evaluation

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when, row_number, rand, lit, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.sql.functions import mean, max
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, NaiveBayes, LinearSVC
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.sql.functions import log, col


# MAX_LEN=12
# MAX_COND_SEQ=56
# MAX_PROC_SEQ=40
# MAX_MED_SEQ=15#37
# MAX_LAB_SEQ=899
# MAX_BMI_SEQ=118


class ML_models():
    def __init__(self,data_icu,k_fold,model_type,concat,oversampling):
        self.data_icu=data_icu
        self.k_fold=k_fold
        self.model_type=model_type
        self.concat=concat
        self.oversampling=oversampling
        self.loss=evaluation.Loss('cpu',True,True,True,True,True,True,True,True,True,True,True)
        self.ml_train()

    def create_kfolds(self):
        """
        Create k-folds for cross-validation

        Returns:
        list containing ids which is in order of folds (ie. 1st k_fold_size/n records are 1st fold, etc.)
        """
            
        labels=pd.read_csv('./data/csv/labels.csv', header=0)
        
        if (self.k_fold==0):
            k_fold=5
            self.k_fold=1
        else:
            k_fold=self.k_fold
        hids=labels.iloc[:,0]
        y=labels.iloc[:,1]
        print("Total Samples",len(hids))
        print("Positive Samples",y.sum())
        #print(len(hids))
        if self.oversampling:
            print("=============OVERSAMPLING===============")
            oversample = RandomOverSampler(sampling_strategy='minority')
            hids=np.asarray(hids).reshape(-1,1)
            hids, y = oversample.fit_resample(hids, y)
            #print(hids.shape)
            hids=hids[:,0]
            print("Total Samples",len(hids))
            print("Positive Samples",y.sum())
        
        ids=range(0,len(hids))
        batch_size=int(len(ids)/k_fold)
        k_hids=[]
        for i in range(0,k_fold):
            rids = random.sample(ids, batch_size)
            ids = list(set(ids)-set(rids))
            if i==0:
                k_hids.append(hids[rids])             
            else:
                k_hids.append(hids[rids])
        return k_hids
    

    # def create_kfolds_spark(self):
    #     """
    #     Create k-folds for cross-validation using PySpark.

    #     Parameters:
    #     k_fold: Number of folds for cross-validation.
    #     oversampling: Boolean indicating whether to apply oversampling for minority class.

    #     Returns:
    #     List of k DataFrames, each representing a fold.
    #     """
    #     # Initialize Spark session
    #     spark = SparkSession.builder \
    #         .appName("CreateKFolds") \
    #         .getOrCreate()

    #     # Read labels data
    #     labels = spark.read.csv('./data/csv/labels.csv', header=True, inferSchema=True)

    #     # Extract hids and labels
    #     hids = labels.select('stay_id')
    #     y = labels.select('label')

    #     if (self.k_fold==0):
    #         k_fold=5
    #         self.k_fold=1
    #     else:
    #         k_fold=self.k_fold

    #     print("Total Samples:", hids.count())
    #     print("Positive Samples:", y.filter(col('label') == 1).count())

    #     if self.oversampling:
    #         print("=============OVERSAMPLING===============")
    #         # Convert to Pandas for oversampling
    #         hids_pd = hids.toPandas()
    #         y_pd = y.toPandas()

    #         oversample = RandomOverSampler(sampling_strategy='minority')
    #         hids_pd = np.asarray(hids_pd).reshape(-1,1)
    #         hids_resampled, y_resampled = oversample.fit_resample(hids_pd, y_pd)
    #         hids_resampled = hids_resampled[:,0]

    #         # Convert back to PySpark DataFrame
    #         hids = spark.createDataFrame(hids_resampled, ['stay_id'])
    #         y = spark.createDataFrame(y_resampled, ['label'])

    #         print("Total Samples after Oversampling:", hids.count())
    #         print("Positive Samples after Oversampling:", y.filter(col('label') == 1).count())
    #         y.unpersist()

    #     hids = hids.withColumn("id", monotonically_increasing_id())
    #     ids = hids.select('id')
    #     batch_size = int(ids.count() / k_fold)
    #     k_hids = []
    #     remaining_ids = hids.select('stay_id')
    #     hids.unpersist()
    #     ids.unpersist()
    #     for i in range(0, k_fold):
    #         fraction = batch_size / remaining_ids.count() if remaining_ids.count() > 0 else 1
    #         if fraction > 1:
    #           sampled_ids = remaining_ids.toPandas()['stay_id']
    #           k_hids.append(sampled_ids)
    #           del sampled_ids
    #         else:
    #           sampled_ids = remaining_ids.sample(fraction=fraction, seed=random.randint(0, 100))
    #           remaining_ids = remaining_ids.subtract(sampled_ids)
    #           sampled_ids = sampled_ids.toPandas()['stay_id']
    #           k_hids.append(sampled_ids)
    #           del sampled_ids

    #     spark.stop()
    #     return k_hids

    def ml_train(self):
        """
        Prepare data for training machine learning models
        """
        k_hids=self.create_kfolds()
        
        labels=pd.read_csv('./data/csv/labels.csv', header=0)
        for i in range(self.k_fold):
            print("==================={0:2d} FOLD=====================".format(i))
            test_hids=k_hids[i]
            train_ids=list(set([0,1,2,3,4])-set([i]))
            train_hids=[]
            for j in train_ids:
                train_hids.extend(k_hids[j])                    
            
            concat_cols=[]
            if(self.concat):
                dyn=pd.read_csv('./data/csv/'+str(train_hids[0])+'/dynamic.csv',header=[0,1])
                dyn.columns=dyn.columns.droplevel(0)
                cols=dyn.columns
                time=dyn.shape[0]

                for t in range(time):
                    cols_t = [x + "_"+str(t) for x in cols]

                    concat_cols.extend(cols_t)
                    
            print('train_hids',len(train_hids))
            X_train,Y_train=self.getXY(train_hids,labels,concat_cols)
            #encoding categorical
            gen_encoder = LabelEncoder()
            eth_encoder = LabelEncoder()
            ins_encoder = LabelEncoder()
            age_encoder = LabelEncoder()
            gen_encoder.fit(X_train['gender'])
          #  eth_encoder.fit(X_train['ethnicity'])
            X_test,Y_test=self.getXY(test_hids,labels,concat_cols)
            all_ethnicities = pd.concat([X_train['ethnicity'], X_test['ethnicity']]).unique()
            #have to fit ethnicity coder on both train and test to avoid errors
            #from ethnicities that are in test but not in train
            eth_encoder.fit(all_ethnicities)
            ins_encoder.fit(X_train['insurance'])
            #age_encoder.fit(X_train['Age'])
            X_train['gender']=gen_encoder.transform(X_train['gender'])
            X_train['ethnicity']=eth_encoder.transform(X_train['ethnicity'])
            X_train['insurance']=ins_encoder.transform(X_train['insurance'])
            #X_train['Age']=age_encoder.transform(X_train['Age'])

            print(X_train.shape)
            print(Y_train.shape)
            print('test_hids',len(test_hids))
            #X_test,Y_test=self.getXY(test_hids,labels,concat_cols)
            self.test_data=X_test.copy(deep=True)
            X_test['gender']=gen_encoder.transform(X_test['gender'])
            X_test['ethnicity']=eth_encoder.transform(X_test['ethnicity'])
            X_test['insurance']=ins_encoder.transform(X_test['insurance'])
            #X_test['Age']=age_encoder.transform(X_test['Age'])
            
            
            print(X_test.shape)
            print(Y_test.shape)
            #print("just before training")
            #print(X_test.head())
            self.train_model_spark(X_train,Y_train,X_test,Y_test)

#Shailesh
    # def ml_train_spark(self):
    #     """
    #     Train machine learning models using PySpark for distributed processing.
    #     """

    #     # Create k-folds using PySpark
    #     k_hids = self.create_kfolds_spark()

    #     spark = SparkSession.builder.appName("ml_train").getOrCreate()
        
    #     # Read labels data
    #     labels=pd.read_csv('./data/csv/labels.csv', header=0)


    #     for i in range(self.k_fold):
    #         print("==================={0:2d} FOLD=====================".format(i))

    #         # Split data into training and testing sets
    #         test_hids=k_hids[i]
    #         train_ids=list(set([0,1,2,3,4])-set([i]))
    #         train_hids=[]
    #         for j in train_ids:
    #             train_hids.extend(k_hids[j])
    #         # Prepare training and testing data
    #         concat_cols = []
    #         if self.concat:
    #             dyn=pd.read_csv('./data/csv/'+str(train_hids[0])+'/dynamic.csv',header=[0,1])
    #             dyn.columns = dyn.columns.droplevel(0)
    #             cols = dyn.columns
    #             time = dyn.shape[0]

    #             for t in range(time):
    #                 cols_t = [x + "_" + str(t) for x in cols]
    #                 concat_cols.extend(cols_t)

    #             continue

    #         train_data = spark.createDataFrame([(h,) for h in train_hids], ["id"])
    #         test_data = spark.createDataFrame([(h,) for h in test_hids], ["id"])

    #         print('Train Data Size:', train_data.count())
            
    #         X_train, Y_train = self.getXY(train_hids, labels, concat_cols)

    #         schema = StructType([
    #             StructField("label", IntegerType(), True)  # Column name, Type, Nullable=True
    #         ])
    #         Y_train = Y_train.to_frame(name="label")

    #         X_train = spark.createDataFrame(X_train)
    #         Y_train = spark.createDataFrame(Y_train, schema=schema)

    #         # Get shape of X_train
    #         print(f"X_train Shape: ({X_train.count()}, {len(X_train.columns)})")

    #         # Get shape of Y_train
    #         print(f"Y_train Shape: ({Y_train.count()}, {len(Y_train.columns)})")

    #         print('Test Data Size:', test_data.count())
    #         X_test, Y_test = self.getXY(test_hids, labels, concat_cols)

    #         self.test_data = X_test
    #         Y_test = Y_test.to_frame(name="label")

    #         X_test = spark.createDataFrame(X_test)
    #         Y_test = spark.createDataFrame(Y_test, schema=schema)

    #         # Get shape of X_train
    #         print(f"X_test Shape: ({X_test.count()}, {len(X_test.columns)})")

    #         # Get shape of Y_train
    #         print(f"Y_test Shape: ({Y_test.count()}, {len(Y_test.columns)})")

    #         # Train the model
    #         self.train_model_spark(X_train, Y_train, X_test, Y_test)

    def train_model(self,X_train,Y_train,X_test,Y_test):
        #logits=[]
        print("===============MODEL TRAINING===============")
        if self.model_type=='Gradient Boosting':
            model = HistGradientBoostingClassifier(categorical_features=[X_train.shape[1]-3,X_train.shape[1]-2,X_train.shape[1]-1]).fit(X_train, Y_train)
            
            prob=model.predict_proba(X_test)
            logits=np.log2(prob[:,1]/prob[:,0])
            self.loss(prob[:,1],np.asarray(Y_test),logits,False,True)
            self.save_output(Y_test,prob[:,1],logits)
        
        elif self.model_type=='Logistic Regression':
            X_train=pd.get_dummies(X_train,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            X_test=pd.get_dummies(X_test,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            
            model = LogisticRegression().fit(X_train, Y_train) 
            logits=model.predict_log_proba(X_test)
            prob=model.predict_proba(X_test)
            self.loss(prob[:,1],np.asarray(Y_test),logits[:,1],False,True)
            self.save_outputImp(Y_test,prob[:,1],logits[:,1],model.coef_[0],X_train.columns)
        
        elif self.model_type=='Random Forest':
            X_train=pd.get_dummies(X_train,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            X_test=pd.get_dummies(X_test,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            model = RandomForestClassifier().fit(X_train, Y_train)
            logits=model.predict_log_proba(X_test)
            prob=model.predict_proba(X_test)
            self.loss(prob[:,1],np.asarray(Y_test),logits[:,1],False,True)
            self.save_outputImp(Y_test,prob[:,1],logits[:,1],model.feature_importances_,X_train.columns)
        
        elif self.model_type=='Xgboost':
            X_train=pd.get_dummies(X_train,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            X_test=pd.get_dummies(X_test,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            model = xgb.XGBClassifier(objective="binary:logistic").fit(X_train, Y_train)
            #logits=model.predict_log_proba(X_test)
            #print(self.test_data['ethnicity'])
            #print(self.test_data.shape)
            #print(self.test_data.head())
            prob=model.predict_proba(X_test)
            logits=np.log2(prob[:,1]/prob[:,0])
            self.loss(prob[:,1],np.asarray(Y_test),logits,False,True)
            self.save_outputImp(Y_test,prob[:,1],logits,model.feature_importances_,X_train.columns)

#Shailesh
    def train_model_spark(self, X_train, Y_train, X_test, Y_test):
        """
        Train machine learning models using PySpark for distributed processing.

        Parameters:
        X_train: PySpark DataFrame containing training features.
        Y_train: PySpark DataFrame containing training labels.
        X_test: PySpark DataFrame containing testing features.
        Y_test: PySpark DataFrame containing testing labels.
        """
        print("===============MODEL TRAINING===============")

        # Combine features and labels into a single DataFrame
        spark = SparkSession.builder.appName("train_model").getOrCreate()
        X_train = X_train.astype('float64')
        Y_train = Y_train.astype('float64').to_frame()
        X_test = X_test.astype('float64')
        Y_test = Y_test.astype('float64').to_frame()
        X_train = X_train.reset_index(drop=True)
        Y_train = Y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        Y_test = Y_test.reset_index(drop=True)
        train_data = pd.concat([X_train, Y_train], axis=1)
        test_data = pd.concat([X_test, Y_test], axis=1)
        # X_train = spark.createDataFrame(X_train)
        # Y_train = spark.createDataFrame(Y_train)
        # X_test = spark.createDataFrame(X_test)
        # Y_test = spark.createDataFrame(Y_test)
        train_data = spark.createDataFrame(train_data)
        test_data = spark.createDataFrame(test_data)

        # labels_train = Y_train.collect()[0][0]
        # labels_test = Y_test.collect()[0][0]
        # train_data = X_train.withColumn('label', lit(labels_train))
        # test_data = X_test.withColumn('label', lit(labels_test))


        # train_data = X_train.join(Y_train, on='id', how='inner')
        # test_data = X_test.join(Y_test, on='id', how='inner')

        # # Encode categorical features
        # for col in categorical_cols:
        #     indexer = StringIndexer(inputCol=col, outputCol=col)
        #     train_data = indexer.fit(train_data).transform(train_data)
        #     test_data = indexer.fit(test_data).transform(test_data)
        #     encoder = OneHotEncoder(inputCol=col, outputCol=col)
        #     train_data = encoder.fit(train_data).transform(train_data)
        #     test_data = encoder.fit(test_data).transform(test_data)

        # feature_cols = [col for col in train_data.columns if col not in ['id', 'label']]
        # assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")


        # Encode categorical features
        categorical_cols = ['gender', 'ethnicity', 'insurance']
      #  indexers = [StringIndexer(inputCol=col, outputCol=f'{col}_index', handleInvalid='skip') for col in categorical_cols]
        encoders = [OneHotEncoder(inputCol=f'{col}', outputCol=f'{col}_encoded') for col in categorical_cols]

        # Assemble features into a single vector
        feature_cols = [col for col in train_data.columns if col not in ['id', 'label'] and col not in categorical_cols] + [f"{col}_encoded" for col in categorical_cols]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        # Define the model
        if self.model_type == 'Logistic Regression':
            model = LogisticRegression(featuresCol="features", labelCol="label")
            pipeline = Pipeline(stages=encoders + [assembler, model])
            trained_model = pipeline.fit(train_data)
            predictions = trained_model.transform(test_data)
            secondProb=F.udf(lambda v:float(v[1]),FloatType())
            predictions = predictions.withColumn("logits", log(secondProb("probability")) /  (1-log(secondProb("probability"))))
            prob = predictions.select("probability").rdd.map(lambda row: row[0][1]).collect()
            logits = predictions.select('logits').rdd.map(lambda row: row[0]).collect()
            logits = np.array(logits, dtype=np.float64)
            prob = np.array(prob, dtype=np.float64)
            Y_test_np = Y_test.values
            self.loss(prob, Y_test_np, logits, False, True)
            #get feature coefficents
            coefficients = trained_model.stages[-1].coefficients.toArray()
            #make sure number of coefficents equal number of input features (Fixing a possible spark bug here)
            feature_assembler = next(stage for stage in trained_model.stages if isinstance(stage, VectorAssembler))
            feature_names = feature_assembler.getInputCols()
            filtered_feature_coefficents = {
                name: coefficent for name, coefficent in zip(feature_names, coefficients[:len(feature_names)])
            }
            coefficients = filtered_feature_coefficents.values()

            self.save_outputImp(Y_test, prob, logits, coefficients, feature_cols)

        elif self.model_type == 'Random Forest':
            model = RandomForestClassifier(featuresCol="features", labelCol="label", rawPredictionCol="rawPrediction")
            pipeline = Pipeline(stages=encoders + [assembler, model])
            trained_model = pipeline.fit(train_data)
            predictions = trained_model.transform(test_data)
            prob = predictions.select("probability").rdd.map(lambda row: row[0][1]).collect()
            secondProb=F.udf(lambda v:float(v[1]),FloatType())
            predictions = predictions.withColumn("logits", log(secondProb("probability")) /  (1-log(secondProb("probability"))))
            prob = predictions.select("probability").rdd.map(lambda row: row[0][1]).collect()
            logits = predictions.select('logits').rdd.map(lambda row: row[0]).collect()
            Y_test_np = Y_test.values
            #get feature importances
            feature_assembler = next(stage for stage in trained_model.stages if isinstance(stage, VectorAssembler))
            feature_names = feature_assembler.getInputCols()
            feature_importances = trained_model.stages[-1].featureImportances.toArray()
            #make sure number of feature importances equal number of input features (Fixing a possible spark bug here)
            filtered_feature_importances = {
                name: importance for name, importance in zip(feature_names, feature_importances[:len(feature_names)])
            }
            feature_importances = filtered_feature_importances.values()
            self.loss(prob, Y_test_np, logits, False, True)
            self.save_outputImp(Y_test, prob, logits, feature_importances, feature_names)
        
        elif self.model_type == 'Gradient Boosting':
            model = GBTClassifier(featuresCol="features", labelCol="label")
            pipeline = Pipeline(stages=encoders + [assembler, model])
            trained_model = pipeline.fit(train_data)
            predictions = trained_model.transform(test_data)
            secondProb=F.udf(lambda v:float(v[1]),FloatType())
            predictions = predictions.withColumn("logits", log(secondProb("probability")) /  (1-log(secondProb("probability"))))
            prob = predictions.select("probability").rdd.map(lambda row: row[0][1]).collect()
            logits = predictions.select('logits').rdd.map(lambda row: row[0]).collect()
            probabilities = predictions.select("probability").rdd.map(lambda row: row[0][1]).collect()
            Y_test_np = Y_test.values
            #get feature importances
            feature_assembler = next(stage for stage in trained_model.stages if isinstance(stage, VectorAssembler))
            feature_names = feature_assembler.getInputCols()
            feature_importances = trained_model.stages[-1].featureImportances.toArray()
            #make sure number of feature importances equal number of input features (Fixing a possible spark bug here)
            filtered_feature_importances = {
                name: importance for name, importance in zip(feature_names, feature_importances[:len(feature_names)])
            }
            feature_importances = filtered_feature_importances.values()
            self.loss(probabilities, Y_test_np, logits, False, True)
            self.save_outputImp(Y_test, probabilities, logits, feature_importances, feature_names)

        elif self.model_type == 'Naive Bayes':
            model = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")
            pipeline = Pipeline(stages=encoders + [assembler, model])
            trained_model = pipeline.fit(train_data)
            predictions = trained_model.transform(test_data)
            secondProb = F.udf(lambda v: float(v[1]), FloatType())
            predictions = predictions.withColumn("logits", log(secondProb("probability")) /  (1-log(secondProb("probability"))))
            probs = predictions.select("probability").rdd.map(lambda row: row[0][1]).collect()
            logits = predictions.select('logits').rdd.map(lambda row: row[0]).collect()
            probs = np.array(probs, dtype=np.float64)
            logits = np.array(logits, dtype=np.float64)
            Y_test_np = Y_test.values
            self.loss(probs, Y_test_np, logits, False, True)
            self.save_output(Y_test, probs, logits)

        elif self.model_type == 'SVM':
            #Max iter set to 100 here for the sake of performance
            model = LinearSVC(featuresCol="features", labelCol="label", maxIter=100)
            pipeline = Pipeline(stages=encoders + [assembler, model])
            trained_model = pipeline.fit(train_data)
            predictions = trained_model.transform(test_data)
            #raw_to_float = F.udf(lambda v: float(v), FloatType())
            predictions = predictions.withColumn("logits", col("rawPrediction"))
            #there is no probability columnm for SVM, so we will use the sigmoid function to approximate them
            logits = predictions.select('logits').rdd.map(lambda row: row[0][1]).collect()
            logits = np.array(logits, dtype=np.float64)
            prob = 1 / (1 + np.exp(-logits))
            Y_test_np = Y_test.values
            self.loss(prob, Y_test_np, logits, False, True)
            self.save_output(Y_test, prob, logits)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def getXY(self,ids,labels,concat_cols):
        X_df=pd.DataFrame()   
        y_df=pd.DataFrame()   
        features=[]
        #print(ids)
        for sample in ids:
            if self.data_icu:
                y=labels[labels['stay_id']==sample]['label']
            else:
                y=labels[labels['hadm_id']==sample]['label']
            
          #  print(sample)
            dyn=pd.read_csv('./data/csv/'+str(sample)+'/dynamic.csv',header=[0,1])
            
            if self.concat:
                dyn.columns=dyn.columns.droplevel(0)
                dyn=dyn.to_numpy()
                dyn=dyn.reshape(1,-1)
                #print(dyn.shape)
                #print(len(concat_cols))
                dyn_df=pd.DataFrame(data=dyn,columns=concat_cols)
                features=concat_cols
            else:
                dyn_df=pd.DataFrame()
                #print(dyn)
                for key in dyn.columns.levels[0]:
                    #print(sample)                    
                    dyn_temp=dyn[key]
                    if self.data_icu:
                        if ((key=="CHART") or (key=="MEDS")):
                            agg=dyn_temp.aggregate("mean")
                            agg=agg.reset_index()
                        else:
                            agg=dyn_temp.aggregate("max")
                            agg=agg.reset_index()
                    else:
                        if ((key=="LAB") or (key=="MEDS")):
                            agg=dyn_temp.aggregate("mean")
                            agg=agg.reset_index()
                        else:
                            agg=dyn_temp.aggregate("max")
                            agg=agg.reset_index()
                    if dyn_df.empty:
                        dyn_df=agg
                    else:
                        dyn_df=pd.concat([dyn_df,agg],axis=0)
                #dyn_df=dyn_df.drop(index=(0))
#                 print(dyn_df.shape)
#                 print(dyn_df.head())
                dyn_df=dyn_df.T
                dyn_df.columns = dyn_df.iloc[0]
                dyn_df=dyn_df.iloc[1:,:]
                        
#             print(dyn.shape)
#             print(dyn_df.shape)
#             print(dyn_df.head())
            stat=pd.read_csv('./data/csv/'+str(sample)+'/static.csv',header=[0,1])
            # print(stat)
            try:
                stat=stat['COND']
                demo=pd.read_csv('./data/csv/'+str(sample)+'/demo.csv',header=0)
#             print(demo.shape)
#             print(demo.head())
                if X_df.empty:
                    X_df=pd.concat([dyn_df,stat],axis=1)
                    X_df=pd.concat([X_df,demo],axis=1)
                else:
                    X_df=pd.concat([X_df,pd.concat([pd.concat([dyn_df,stat],axis=1),demo],axis=1)],axis=0)
                if y_df.empty:
                    y_df=y
                else:
                    y_df=pd.concat([y_df,y],axis=0)
            except:
    #             print(stat.shape)
    #             print(stat.head())
                demo=pd.read_csv('./data/csv/'+str(sample)+'/demo.csv',header=0)
    #             print(demo.shape)
    #             print(demo.head())
                if X_df.empty:
                    X_df=dyn_df
                    X_df=pd.concat([X_df,demo],axis=1)
                else:
                    X_df=pd.concat([X_df,pd.concat([dyn_df, demo],axis=1)],axis=0)
                if y_df.empty:
                    y_df=y
                else:
                    y_df=pd.concat([y_df,y],axis=0)
#             print("X_df",X_df.shape)
#             print("y_df",y_df.shape)
        # print("X_df",X_df.shape)
        # print("y_df",y_df.shape)
        return X_df ,y_df
    

    #decided not to convert to spark as I would have to convert back to pandas for
    #pickel dumping -Josh
    def save_output(self,labels,prob,logits):
        output_df=pd.DataFrame()
        output_df['Labels']=labels
        output_df['Prob']=prob
        output_df['Logits']=np.asarray(logits)
        output_df['ethnicity']=list(self.test_data['ethnicity'])
        output_df['gender']=list(self.test_data['gender'])
        output_df['age']=list(self.test_data['Age'])
        output_df['insurance']=list(self.test_data['insurance'])
        
        with open('./data/output/'+'outputDict', 'wb') as fp:
               pickle.dump(output_df, fp)
        
    
    #decided not to convert to spark as I would have to convert back to pandas for
    #pickel dumping and for csv writing -Josh
    def save_outputImp(self,labels,prob,logits,importance,features):
        output_df=pd.DataFrame()
        output_df['Labels']=labels
        output_df['Prob']=prob
        output_df['Logits']=np.asarray(logits)
        output_df['ethnicity']=list(self.test_data['ethnicity'])
        output_df['gender']=list(self.test_data['gender'])
        output_df['age']=list(self.test_data['Age'])
        output_df['insurance']=list(self.test_data['insurance'])
        
        with open('./data/output/'+'outputDict', 'wb') as fp:
               pickle.dump(output_df, fp)
        
        imp_df=pd.DataFrame()
        imp_df['imp']=importance
        imp_df['feature']=features
        imp_df.to_csv('./data/output/'+'feature_importance.csv', index=False)
                
                



    # def getXY_spark(self,ids,labels,concat_cols):
    #     spark = SparkSession.builder.appName("getXY").getOrCreate()
    #     X_df = pd.DataFrame()
    #     y_df = pd.DataFrame()
    #     features = []
    #     #labels = spark.createDataFrame(labels)
    #     X_is_empty = True
    #     y_df_is_empty = True
    #     for row in ids.rdd.toLocalIterator():
    #         sample = row['id']
    #         if self.data_icu:
    #             y = labels.filter(col('stay_id')==sample).select('label')
    #         else:
    #             y = labels.filter(col('stay_id')==sample).select('label')
    #         dyn=pd.read_csv('./data/csv/'+str(sample)+'/dynamic.csv',header=[0,1])
    #         if self.concat:
    #             dyn.columns=dyn.columns.droplevel(0)
    #             dyn=dyn.to_numpy()
    #             dyn=dyn.reshape(1,-1)
    #             #print(dyn.shape)
    #             #print(len(concat_cols))
    #             dyn_df=pd.DataFrame(data=dyn,columns=concat_cols)
    #             features = concat_cols
    #         else:
    #             dyn_df=pd.DataFrame()
    #             is_empty = True
    #             for key in dyn.columns.levels[0]:
    #                 dyn_temp = dyn[key]
    #                 dyn_temp = spark.createDataFrame(dyn_temp.to_frame())
    #                 if self.data_icu:
    #                     if ((key=="CHART") or (key=="MEDS")):
    #                         agg = dyn_temp.agg(mean('*'))
    #                     else:
    #                         agg = dyn_temp.agg(max('*'))
    #                 else:
    #                     if ((key=="LAB") or (key=="MEDS")):
    #                         agg = dyn_temp.agg(mean('*'))
    #                     else:
    #                         agg = dyn_temp.agg(max('*'))
    #                 if is_empty:
    #                     dyn_df = agg
    #                     is_empty = False
    #                 else:
    #                     dyn_df = dyn_df.union(agg)
    #             dyn_df = dyn_df.transpose()
    #             first_row = dyn_df.head()
    #             new_columns = [str(x) for x in first_row]
    #             dyn_df = dyn_df.toDF(*new_columns)

    #       #  stat=pd.read_csv('./data/csv/'+str(sample)+'/static.csv',header=[0,1])
    #        # print(stat)
    #         #stat = stat['Cond']
    #        # stat = spark.createDataFrame(stat)
    #         demo = spark.read.csv(f'./data/csv/{sample}/demo.csv', header=True, inferSchema=True)
    #         if X_is_empty:
    #          #   X_df = dyn_df.join(stat)d
    #             X_df = spark.createDataFrame(dyn_df)
    #             X_df = X_df.join(demo)
    #             X_is_empty = False
    #         else:
    #            # temp_df = dyn_df.join(stat)
    #             temp_df = spark.createDataFrame(dyn_df)
    #             temp_df = temp_df.join(demo)
    #             X_df = X_df.union(temp_df)
    #         if y_df_is_empty:
    #             y_df = y
    #             y_df_is_empty = False
    #         else:
    #             y_df = y_df.union(y)
                
    #     print(X_df.show())
    #     num_rows = X_df.count()
    #     num_columns = len(X_df.columns)
    #     print(f'X_df ({num_rows}, {num_columns})')
    #     num_rows = y_df.count()
    #     num_columns = len(y_df.columns)
    #     print(f'y_df ({num_rows}, {num_columns})')
    #     spark.stop()
    #     return X_df, y_df











                
            




