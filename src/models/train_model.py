from sklearn.pipeline import Pipeline
import xgboost as xgb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import lightgbm as lgb

def build_xgb_modelling_pipeline(preprocessor, param_grid):
    """
    Function to build a pipeline which does the preprocessing steps and 
    then instantiates and xgboost regressor model
    """

    # Pipeline to preprocess the data and instantiate a XGBRegressor model
    xgb_pipe = Pipeline(
        steps = [
            ('preprocessor', preprocessor),
            ('xgb_regressor', xgb.XGBRegressor(objective='reg:squarederror', 
                                                n_estimators=param_grid['n_estimators'],
                                                learning_rate=param_grid['learning_rate'],
                                                max_depth=param_grid['max_depth'],
                                                min_child_weight=param_grid['min_child_weight'],
                                                gamma=param_grid['gamma'],
                                                subsample=param_grid['subsample'],
                                                colsample_bytree=param_grid['colsample_bytree'],
                                                random_state=42))
        ]
    )

    return xgb_pipe


def train_xgb_model(xgb_pipe, x_train, y_train):
    """
    Function that trains an xgboost model
    """
    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    # Fit the pipeline to training data
    xgb_pipe.fit(x_train, 
                y_train)
    
    return xgb_pipe


def plot_feature_importance_xgb_model(xgb_pipe, preprocessor):
    """
    Function to find out the feature importances and plot them
    in a bar plot
    """
    # XGBoost Regressor model
    xgb_model = xgb_pipe.named_steps['xgb_regressor']
    # feature names
    feature_names = preprocessor.get_feature_names_out()
    # feature importances
    feature_importance = xgb_model.feature_importances_

    df = pd.DataFrame({
        'feature_name': feature_names,
        'feature_importance': feature_importance
    })

    high_cardinal_features = {
        'high_cardinal_cats__0': 'high_cardinal_cats__departure_date',
        'high_cardinal_cats__1': 'high_cardinal_cats__days_from_flight'
    }

    df['feature_name'] = df['feature_name'].replace(high_cardinal_features)

    df = df.sort_values(by='feature_importance', ascending=False)

    # Create a bar plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='feature_importance', y='feature_name', data=df)

    # Customize the plot
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Feature Importances')
    plt.show()


def create_lgbm_pipeline(preprocessor, X_train, y_train, params):
    
    
    
    lgb_pipe = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('lgb', lgb.LGBMRegressor(**params))
        ]
    )
    
    lgb_pipe.fit(X_train, y_train)
    return lgb_pipe



def create_xgb_pipeline(preprocessor, X_train, y_train, params):
    xgb_pipe = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('xgb', xgb.XGBRegressor(**params))
        ]
    )
    xgb_pipe.fit(X_train, y_train)
    return xgb_pipe
