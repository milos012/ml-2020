import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    df = pd.read_csv('loan database/train_u6lujuX_CVtuZ9i.csv')

    #Pregled skupa za testiranje

    #print(df.shape)
    #print(df.head())
    #print(df.info())
    #print(df.describe())
    #print(df.duplicated().any())
    #print('The percentage of Y class : %.2f' % (df['Loan_Status'].value_counts()[0] / len(df)))
    #print('The percentage of N class : %.2f' % (df['Loan_Status'].value_counts()[1] / len(df)))

    #Izbacujemo id kolonu iz tabele jer nam nije potrebna
    df.drop('Loan_ID', axis=1, inplace=True)



    #Credit_History
    grid = sns.FacetGrid(df, col='Loan_Status', size=3, aspect=1.5)
    grid.map(sns.countplot, 'Credit_History')
    #plt.show(grid)

    # na osnovu grafa vidimo da je kredit odobren vecini ljudi kojima je Credit History = 1, takodje broj ljudi
    # kojima je odobren kredit a Credit History = 0 je veoma mali.
    # Zakljucak : ako je Credit History = 1 , veca je sansa za kredit

    #Gender
    grid = sns.FacetGrid(df, col='Loan_Status', size=3, aspect=1.5)
    grid.map(sns.countplot, 'Gender')
    #plt.show(grid)

    #I muskarci i zene dobijaju odobrenja za kredit tako da ovaj podatak nije bitan

    #Married
    #plt.figure(figsize=(10, 5))
    #plt.show(sns.countplot(x='Married', hue='Loan_Status', data=df))

    #Vidimo da ljudi koji su u braku imaju vecu sansu za kredit

    #Dependents
    #plt.figure(figsize=(10, 5))
    #plt.show(sns.countplot(x='Dependents', hue='Loan_Status', data=df))

    #Ako je Dependents=0 ili 2 veca je sansa da se dobije odobrenje za kredit nego ako je Dependents=1 ili 3

    #Education
    grid = sns.FacetGrid(df, col='Loan_Status', size=3, aspect=1.5)
    grid.map(sns.countplot, 'Education')
    #plt.show(grid)

    #i Graduate i Not Graduate dobijaju odobrenje za kredit => nebitan podatak

    #Self_Employed
    grid = sns.FacetGrid(df, col='Loan_Status', size=3, aspect=1.5)
    grid.map(sns.countplot, 'Self_Employed')
    #plt.show(grid)

    #Ista situacija kao i kod Education => nebitan podatak

    #Property_Area
    plt.figure(figsize=(15, 5))
    plt.show(sns.countplot(x='Property_Area', hue='Loan_Status', data=df))

    #Ljudi koji zive u Semiurban sredini imaju veoma veliku sansu da dobiju odobrenje za kredit

    #Applicant Income
    plt.show(plt.scatter(df['ApplicantIncome'], df['Loan_Status']))
    #print(df.groupby('Loan_Status').median())

    #sto je CoapplicantIncome veci to je veca sansa za odobrenje kredita


    #Resavanje problema nultih vrednosti
    #Podelicemo podatke na kategoricke i numericke
    kategoricki = []
    numericki = []
    for i, c in enumerate(df.dtypes):
        if c == object:
            kategoricki.append(df.iloc[:, i])
        else:
            numericki.append(df.iloc[:, i])

    kategoricki = pd.DataFrame(kategoricki).transpose()
    numericki = pd.DataFrame(numericki).transpose()
    #print(kategoricki.head())
    #print("  ")
    #print(numericki.head())

    #Popunjavanje nulltih vrednosti sa najpopularnijim podacima
    kategoricki = kategoricki.apply(lambda x: x.fillna(x.value_counts().index[0]))
    #print(kategoricki.isnull().sum().any())

    #Popunjavanje nultih vrednosti vrednostima njihovih prethodnika
    numericki.fillna(method='bfill', inplace=True)
    #print(numericki.isnull().sum().any())