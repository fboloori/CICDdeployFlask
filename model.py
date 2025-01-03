# importing required libraries 
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier 

class MLmodel:

    def __init__(self):
            self.scaler= StandardScaler()
            heart = pd.read_csv("heart_cleveland_upload.csv")
            heart_df = heart.copy()
            heart_df = heart_df.rename(columns={'condition':'target'})

            x= heart_df.drop(columns= 'target')
            y= heart_df.target
            x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=42)

            x_train_scaler= self.scaler.fit_transform(x_train)
            x_test_scaler= self.scaler.fit_transform(x_test)

            self.model=RandomForestClassifier(n_estimators=20)
            self.model.fit(x_train_scaler, y_train)

            y_pred= self.model.predict(x_test_scaler)
            self.score = self.model.score(x_test_scaler,y_test)
            self.cm = confusion_matrix(y_test, y_pred)

            print(self.score)
            print('Classification Report\n', classification_report(y_test, y_pred))
            print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))
            print(self.cm)

            # Creating a pickle file for the classifier
            self.modelfilename = 'model.pkl'
            pickle.dump(self.model, open(self.modelfilename, 'wb'))
            print('pickle file saved. ')   

    def predict(self , newx):
            newx_scaler= self.scaler.fit_transform(newx)
            with open(r"model.pkl", "rb") as input_file:
                model = pickle.load(input_file)
                
            res = np.array( model.predict(newx) ,dtype = float) 
            return (res)



