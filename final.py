import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def dimention_reduction(method, x, feature_num=0, y=None):
    '''
        pca: 
            unsupervised mode, thus only x is utilized
        lda: 
            1. supervised mode, thus both x,y is utilized
            2. limit : n_component < N_class
    '''

    if method == "pca":
        return pca_transform(x, feature_num)
    elif method == "lda":
        return lda_transform(x, y, feature_num)
    else:
        print("no dimension reduction")
        return x


def lda_transform(x, y, feature_num):
    '''
        when train mode: lda_model is fut with training data, and save as global var
        when test mode: access lda model save as global var 
    '''
    global lda_model
    # train mode
    if y is not None:
        lda_model = LDA(n_components=feature_num)
        lda_model = lda_model.fit(x, y)
        return lda_model.transform(x)
    # test mode
    else:
        return lda_model.transform(x)


def pca_transform(x, feature_num):

    # normalization
    scaler = StandardScaler()
    scaler.fit(x)
    scaled_data = scaler.transform(x)

    # 第三步: 對數據做PCA降維，參數n_components表示降低至多少維度

    pca = PCA(n_components=feature_num)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)

    return x_pca


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
        elif col_name in ['Count', 'Country', 'State', 'City', 'Quarter']:
            continue
        elif df[col_name].dtypes == 'float64':
            # store the mean
            mapper[col_name] = df[col_name].mean()
        else:
            # store the categories and the mode
            categories = df[col_name].unique()
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
        elif col_name in ['Count', 'Country', 'State', 'City', 'Quarter']:  # useless features
            df = df.drop(col_name, axis=1)
        elif df[col_name].dtypes == 'float64':  # numeric features
            val = mapper[col_name]
            df[col_name] = df[col_name].fillna(val)
        else:  # categorical features
            categories, val = mapper[col_name]
            df[col_name] = df[col_name].fillna(val)
            # encode categorical features to one-hot
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


DATA_DIR = './html2021final/'
OUT_DIR = './output/'


def main():
    warnings.filterwarnings('ignore')
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    seed = 0  # for reproducibility
    
    # set dimention reduction type and set number of features
    global dim_reduce_type
    dim_reduce_type = "lda"
    feature_num = 5
    if dim_reduce_type == "lda":
        # cause feature_num is requested < N_class in lda
        feature_num = 5
    elif dim_reduce_type == "pca":
        feature_num = 50
    # training phase
    customers = load_data(DATA_DIR, mode='Train')
    mapper = get_mapper(customers)
    x, y, x_unlabeled = preprocess_data(customers, mapper)

    print(f'train data shape: {x.shape}, {y.shape}')
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=seed)

    # pca or lda for dimention reduction
    x_train = dimention_reduction(method=dim_reduce_type, x=x_train, feature_num=feature_num, y=y_train)
    x_val = dimention_reduction(method=dim_reduce_type, x=x_val, feature_num=feature_num, y=y_val)
    x_unlabeled = dimention_reduction(method=dim_reduce_type, x=x_unlabeled, feature_num=feature_num)
    
    model = train(x_train, x_val, y_train, y_val, random_state=seed)

    print(f'unlabeled data shape (before filtering): {x_unlabeled.shape}')
    x_unlabeled = filter_data_by_confidence(x_unlabeled, model, threshold=0.9)

    print(f'unlabeled data shape (after filtering): {x_unlabeled.shape}')

    y_unlabeled = model.predict(x_unlabeled)
    x_train = np.concatenate((x_train, x_unlabeled), axis=0)
    y_train = np.concatenate((y_train, y_unlabeled), axis=0)
    model = train(x_train, x_val, y_train, y_val, random_state=seed)

    # testing phase
    customers = load_data(DATA_DIR, mode='Test')
    _, _, x_test = preprocess_data(customers, mapper)

    x_test = dimention_reduction(
        method=dim_reduce_type, x=x_test, feature_num=feature_num)

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
