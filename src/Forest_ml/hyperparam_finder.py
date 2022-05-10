
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold






def finder(X_train,y_train,model_num:int):

    if model_num == 0:
        model = LogisticRegression(multi_class='multinomial')
        """
        p_grid = {
        'random_state':[54],
        'max_iter':[500],
        'C':[0.1]
        }
        """
        p_grid = {
        'random_state':[1,2,3,4,56],
        'max_iter':[1000,10000],
        'C':[0.1,1,2,0.001,0.0001]
        }
    else:
        model = RandomForestClassifier()
        p_grid = {
        'criterion':['gini', 'entropy'],
        'max_depth':[10,100,1000,10000,500],
        'n_estimators':[100,200,300,400,500,800],
        }

    inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=42)

    # Nested CV with parameter optimization
    # clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv)
    # nested_score = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv)
        

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=outer_cv)
    clf.fit(X_train, y_train)

    return  clf.best_params_,clf.best_score_








