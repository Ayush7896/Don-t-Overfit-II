import csv
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.preprocessing import RobustScaler,FunctionTransformer,PowerTransformer
import pickle
from matplotlib import image, pyplot as plt
import seaborn as sns
from PIL import Image
import time
import scipy.stats as stat
import pylab
import random

st.set_page_config(page_title="Don't Overfit II",page_icon="")

df_train=pd.read_csv("train.csv")
data_copy=df_train.drop(['id'],axis=1)


def main():
    st.title("Don't Overfit II")
    st.write("Long ago, in the distant, fragrant mists of time, there was a competition.It was not just any competition.It was a competition that challenged mere mortals to model a 20,000x200 matrix of continuous variables using only 250 training samples without overfitting.")

    @st.cache
    def modelling():

        with open('model_ro.pkl','rb') as file1:
            robust_scaler=pickle.load(file1)

        with open('model_boxcox.pkl','rb') as file2:
            box_cox=pickle.load(file2)
            
        with open('model_calib.pkl','rb') as file3:
            model=pickle.load(file3)

        with open('dictionary_values.pkl','rb') as file4:
            dict_values=pickle.load(file4)

        return robust_scaler,box_cox,model,dict_values

    robust_scaler,box_cox,model,dict_values=modelling()

    threshold=0.5609943982086608

    def main_function(X):
        br=0
        if sum(X.isna().sum().values.tolist())>0:
              nan_cols = [i for i in X.columns if X[i].isnull().any()]
              st.write("You have some missing value features,the feature values are",nan_cols)
              br=1 
        else:
              pass

        if br==0:
            for key,value in dict_values.items():
                if X[str(key)].values[0]>value[0]:
                   st.write("Error the feature number", key,"is excedding the maximum value by this",X[str(key)].values[0]-value[0])
                   st.write("The maximum value is",value[0])
                   st.write("the minimum value is ",value[1])
                   st.write("the value should be in between",value[1] ,"and",value[0])
                    
                   br=1
               

                if X[str(key)].values[0]<value[1]:
                    st.write("error the feature number",key ,"is exceeding the minimum value by this",value[1]-X[str(key)].values[0])
                    br=1

        if br==0:  

            X=robust_scaler.transform(X)
            X=box_cox.transform(X)
            pred=model.predict_proba(X)[:,1]
            if pred>=threshold:
                return 1
            else:
                return 0
    csv1=('csv1.csv')
    csv2=('csv2.csv')
    csv3=('csv3.csv')
    csv4=('csv4.csv')
    csv_list=[csv1,csv2,csv3,csv4]
    choices = st.radio(
     "What's your choice",
     ('Existing CSV', 'Upload a CSV'))

    if  choices == 'Existing CSV':
        option = st.selectbox(
            'Choose CSV File',
            (csv1,csv2,csv3,csv4))
        st.write('You selected:', option)

        data_sample=pd.read_csv(option)
        df=st.dataframe(data_sample)
    else:
        st.caption("Please upload a csv of 300 features with min and max range of -1 and +2")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data_sample = pd.read_csv(uploaded_file)
            data_sample=pd.DataFrame(data_sample)
            st.write(data_sample)

    if st.button("Click to Predict"):
        with st.spinner('Wait for it...'):
            time.sleep(2)
        prediction=main_function(data_sample)
        
        st.caption("Predicted Value:")
        st.text(prediction)


    image=Image.open("Image.png")
    st.image(image,caption="Kaggle Competition Score",output_format="png")   


    with st.sidebar.header("this is sidebar"):
        sns.histplot(x="2",data=df_train,kde=True)
        plt.show()
        plt.xlabel("feature")
        st.pyplot()

    with st.sidebar.header("this is sidebar"):
        correlations=data_copy[data_copy.columns[:]].corr()['target']
        positive_correlations=correlations.sort_values(ascending=False)[:20]
        plt.figure(figsize=(8,8))
        sns.barplot(x=positive_correlations.values[1:],y=positive_correlations.index[1:])
        plt.xlabel("correlation magnitude")
        plt.ylabel("features")
        plt.title("Top 20 Features highly positive correlated with target")
        plt.show()
        st.pyplot()


    with st.sidebar.header("this is sidebar"):
        def QQ_plot(data,feature):
            plt.figure(figsize=(12,8))
            plt.subplot(1,2,1)
            sns.histplot(data[feature],kde=True)
            plt.subplot(1,2,2)
            stat.probplot(data[feature],dist='norm',plot=pylab)
            plt.show()
            st.pyplot()
        plot1=QQ_plot(data_copy,data_copy.columns[23])

if __name__=='__main__':
        main()





           