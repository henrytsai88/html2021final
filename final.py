import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN


def lda_transform(x, y, feature_num, mode, random_state):
    '''
        when train mode: lda_model is fut with training data, and save as global var
        when test mode: access lda model save as global var 
    '''
    global lda_model
    # train mode
    if mode == "train":
        lda_model = LDA(n_components=feature_num)
        lda_model = lda_model.fit(x, y)
        return lda_model.transform(x)
    # test mode
    else:
        return lda_model.transform(x)


def pca_transform(x, feature_num, mode, random_state):

    global pca_model, scalar_model

    if mode == "train":
        # normalization
        scalar_model = StandardScaler()
        scalar_model.fit(x)
        scaled_data = scalar_model.transform(x)

        # n_components means resulted dimension
        pca_model = PCA(n_components=feature_num, random_state=random_state)
        x_pca = pca_model.fit(scaled_data)
        x_pca = pca_model.transform(scaled_data)
    else:
        scaled_data = scalar_model.transform(x)
        x_pca = pca_model.transform(scaled_data)
    return x_pca


def dimention_reduction(method, x, feature_num=0, y=None, mode="", random_state=0):
    '''
        pca: 
            unsupervised mode, thus only x is utilized
        lda: 
            1. supervised mode, thus both x,y is utilized
            2. limit : n_component < N_class
    '''

    if method == "pca":
        return pca_transform(x, feature_num, mode, random_state=random_state)
    elif method == "lda":
        return lda_transform(x, y, feature_num, mode, random_state=random_state)
    else:
        print("no dimension reduction")
        return x


def outlier(x, y, threshold, random_state=0):
    '''
        detect outlier in unsupervised way
        decision_function: Average anomaly score, higher means more likely to be outlier
        how: based on path length( number of splittings ) in a tree
            if a sample needs less split, means it's more likely to be outlier
    '''
    clf = IsolationForest(bootstrap=True, random_state=random_state)
    clf.fit(x)
    scores_pred = clf.decision_function(x)
    print('worst outlier score:\t{:.4f}'.format(np.min(scores_pred)))
    indices = np.where(scores_pred > threshold)[0]
    print('keep {:.3f} of data'.format(len(indices)/len(x)))
    
    return x[indices], y[indices]


def resample_data(x, y, random_state=0):
    """
        Over-sampling using SMOTE, and then under-sampling using ENN.
    """
    smote = SMOTEENN(random_state=random_state)
    x, y = smote.fit_resample(x, y)
    return x, y


def filter_data_by_confidence(x, model, threshold):
    """
        Filter data whose predicted confidence is higher than the threshold.
    """
    y_confidence = model.predict_proba(x).max(axis=1)
    y_confidence = np.array(y_confidence)
    indices = np.where(y_confidence >= threshold)[0]
    return x[indices]


def load_data(dir, mode='Train'):
    """
        Load all csv files and combine them to a dataframe.
    """
    # load train/test IDs
    assert mode in ['Train', 'Test']
    filepath = os.path.join(dir, f'{mode}_IDs.csv')
    df = pd.read_csv(filepath)

    # combine dataframes by 'Customer ID'
    for file in ['demographics.csv', 'location.csv', 'satisfaction.csv', 'services.csv', 'status.csv']:
        filepath = os.path.join(dir, file)
        df = pd.merge(df, pd.read_csv(filepath), on='Customer ID', how='left')

    # load 'population.csv'
    filepath = os.path.join(dir, 'population.csv')
    population = pd.read_csv(filepath)

    # map 'Zip Code' to 'Population'
    df['Population'] = df['Zip Code'].replace(
        population.set_index('Zip Code')['Population'])
    df = df.drop('Zip Code', axis=1)

    # replace 'Latitude' and 'Longitude' if 'Lat Long' is not nan
    for index, _ in df.iterrows():
        if not pd.isnull(df.loc[index, 'Lat Long']):
            # extract float numbers from string by regular expression
            latlong = re.findall(r'[-+]?\d*\.\d+|\d+',
                                 df.loc[index, 'Lat Long'])
            df.loc[index, ['Latitude', 'Longitude']] = list(
                map(float, latlong))
    df = df.drop('Lat Long', axis=1)

    return df


def get_mapper(df):
    """
        Compute the mapping functions from data.
        The mapping functions are used for missing value imputation.
    """
    mapper = {}
    for col_name in df.columns:
        if col_name in ['Customer ID', 'Churn Category']:
            continue
        elif col_name in ['Count', 'Count_x', 'Count_y', 'Country', 'State', 'City', 'Quarter']:
            continue
        elif df[col_name].dtypes == 'float64':
            # store the mean
            mapper[col_name] = df[col_name].mean()
        else:
            # store the categories and the mode
            categories = df[col_name].unique()
            categories = categories[~pd.isnull(categories)]
            mapper[col_name] = (categories, df[col_name].mode()[0])
    return mapper


def preprocess_data(df, mapper):
    """
        Preprocess data to numpy arrays, including:
        1. impute missing values,
        2. encode categorical features to one-hot,
        3. split labeled/unlabeled data,
        4. split features/labels.
    """
    print(df['Churn Category'].unique())
    # impute missing values
    # print(f'features before preprocessing:\n{df.columns}\n')
    for col_name in df.columns:
        if col_name in ['Customer ID', 'Churn Category']:  # skip
            continue
        # useless features
        elif col_name in ['Count', 'Count_x', 'Count_y', 'Country', 'State', 'City', 'Quarter']:
            df = df.drop(col_name, axis=1)
        elif df[col_name].dtypes == 'float64':  # numeric features
            val = mapper[col_name]
            df[col_name] = df[col_name].fillna(val)
        else:  # categorical features
            categories, val = mapper[col_name]
            df[col_name] = df[col_name].fillna(val)
            # encode categorical features to one-hot
            if len(categories) == 2:  # binary case (Yes/No, Male/Female)
                df[col_name] = df[col_name].map(
                    lambda x: 1 if x == categories[0] else 0)
            else:
                for category in categories:
                    df[f'{col_name}_{category}'] = df[col_name].map(
                        lambda x: 1 if x == category else 0)
                df = df.drop(col_name, axis=1)
    features = [col_name for col_name in df.columns if col_name not in [
        'Customer ID', 'Churn Category']]
    # print(f'features after preprocessing:\n{features}\n')

    # split labeled/unlabeled data
    labeled_data = df[df['Churn Category'].notna()]
    unlabeled_data = df[df['Churn Category'].isna()]

    # split features/labels, convert to numpy array
    x_labeled = labeled_data[features].to_numpy()
    y_labeled = labeled_data['Churn Category'].to_numpy()
    x_unlabeled = unlabeled_data[features].to_numpy()

    return x_labeled, y_labeled, x_unlabeled


def train(x_train, x_val, y_train, y_val, random_state=0):
    """
        Train the model.
    """
    # train with xgboost
    best_model, best_f1_score = None, 0
    for n_estimators in [10, 20, 50, 100]:
        # regularization for xg_boost
        for gamma in [0, 2]:
            print(f'n_estimators: {n_estimators}, gamma: {gamma}')
            model = XGBClassifier(gamma=gamma, n_estimators=n_estimators,
                                  learning_rate=0.3, verbosity=0, random_state=random_state)
            model.fit(x_train, y_train)

            # print(f'feature importance:\n{model.feature_importances_}')
            plot_importance(model)
            plt.savefig(os.path.join(OUT_DIR, f'{n_estimators}_{gamma}.png'))

            y_pred = model.predict(x_train)
            train_f1_score = f1_score(y_train, y_pred, average='macro')
            print('train f1 score:\t{:.4f}'.format(train_f1_score))

            y_pred = model.predict(x_val)
            val_f1_score = f1_score(y_val, y_pred, average='macro')
            print('val f1 score:\t{:.4f}\n'.format(val_f1_score))

            if val_f1_score > best_f1_score:
                best_model = model
                best_f1_score = val_f1_score

    return best_model

def train_SVC(x_train, x_val, y_train, y_val, random_state=0):
    """
        Train the model.
    """
    
    model = SVC(gamma='auto', degree=1, class_weight='balanced', coef0=14.0, shrinking=True, kernel='rbf', probability=True, C=1e3)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_train)
    train_f1_score = f1_score(y_train, y_pred, average='macro')
    print('train f1 score:\t{:.4f}'.format(train_f1_score))

    y_pred = model.predict(x_val)
    val_f1_score = f1_score(y_val, y_pred, average='macro')
    print('val f1 score:\t{:.4f}\n'.format(val_f1_score))
    
    return model

def train_LogisticRegression(x_train, x_val, y_train, y_val, random_state=0):
    """
        Train the model.
    """
    # train with logistic regression
    model = LogisticRegression(class_weight=None, random_state=random_state, solver='lbfgs')
    model.fit(x_train, y_train)

    y_pred = model.predict(x_train)
    train_f1_score = f1_score(y_train, y_pred, average='macro')
    print('train f1 score:\t{:.4f}'.format(train_f1_score))

    y_pred = model.predict(x_val)
    val_f1_score = f1_score(y_val, y_pred, average='macro')
    print('val f1 score:\t{:.4f}\n'.format(val_f1_score))

    # if val_f1_score > best_f1_score:
    #     best_model = model
    #     best_f1_score = val_f1_score

    return model

DATA_DIR = './html2021final/'
OUT_DIR = './output/'


def main():
    warnings.filterwarnings('ignore')
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    seed = 0  # for reproducibility
    # used for ablation test
    config = {
        'dim_reduction' : True,
        'resample' : False,
        'outlier_removal' : True,
        'pseudo_label' : False
    }
    # training phase
    customers = load_data(DATA_DIR, mode='Train')
    mapper = get_mapper(customers)
    x, y, x_unlabeled = preprocess_data(customers, mapper)

    print(f'train data shape: {x.shape}, {y.shape}')
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=seed)
    if config['outlier_removal']:
        x_train, y_train = outlier(
            x=x_train, y=y_train, threshold=-0.05, random_state=seed)
    if config['resample']:
        print(
            f'train label distribution (before resampling):\n{pd.value_counts(y_train)}')
        x_train, y_train = resample_data(x_train, y_train, random_state=seed)
        print(
            f'train label distribution (after resampling):\n{pd.value_counts(y_train)}')
        
    # set dimention reduction type and set number of features
    if config['dim_reduction']:
        global dim_reduce_type
        dim_reduce_type = "pca"
        feature_num = 5
        if dim_reduce_type == "lda":
            # cause feature_num is requested < N_class in lda
            feature_num = 5
        elif dim_reduce_type == "pca":
            feature_num = 40
        # pca or lda for dimention reduction
        x_train = dimention_reduction(
            method=dim_reduce_type, x=x_train, feature_num=feature_num, y=y_train, mode="train", random_state=seed)
        x_val = dimention_reduction(
            method=dim_reduce_type, x=x_val, feature_num=feature_num, y=y_val, mode="val", random_state=seed)
        x_unlabeled = dimention_reduction(
            method=dim_reduce_type, x=x_unlabeled, feature_num=feature_num, mode="val", random_state=seed)
    
    model = train_SVC(x_train, x_val, y_train, y_val, random_state=seed)

    if config['pseudo_label']:
        print(f'unlabeled data shape (before filtering): {x_unlabeled.shape}')

        x_unlabeled = filter_data_by_confidence(x_unlabeled, model, threshold=0.9)
        y_unlabeled = model.predict(x_unlabeled)

        

        print(f'unlabeled data shape (after filtering): {x_unlabeled.shape}')

        x_train = np.concatenate((x_train, x_unlabeled), axis=0)
        y_train = np.concatenate((y_train, y_unlabeled), axis=0)
        model = train(x_train, x_val, y_train, y_val, random_state=seed)

    # testing phase
    customers = load_data(DATA_DIR, mode='Test')
    _, _, x_test = preprocess_data(customers, mapper)

    if config['dim_reduction']:
        x_test = dimention_reduction(
            method=dim_reduce_type, x=x_test, feature_num=feature_num, mode="test", random_state=seed)

    print(f'test data shape: {x_test.shape}')
    y_pred = model.predict(x_test)
    print(y_pred)

    # output csv file
    out = pd.DataFrame({
        'Customer ID': customers['Customer ID'],
        'Churn Category': y_pred
    })
    # map 'Churn Category' from string format to numeric format
    out['Churn Category'] = out['Churn Category'].map({
        'No Churn': 0, 'Competitor': 1, 'Dissatisfaction': 2, 'Attitude': 3, 'Price': 4, 'Other': 5
    })
    out.to_csv(os.path.join(OUT_DIR, 'submission.csv'), index=False)


if __name__ == '__main__':
    main()
