from sklearn.feature_selection import SelectKBest, f_classif

def select_best_features(X, y, k=10):
    # Select top k features based on ANOVA F-statistic
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    return X_selected
