def train_and_evaluate_svr_with_tuning(X_train_feats, y_train_tgt, X_test_feats, y_test_tgt, target_name, arch_name):
    print(f"\n--- Training SVR for {target_name} ({arch_name}) with Hyperparameter Tuning ---")
    svr = SVR()
    param_grid = {
        'kernel': ['rbf', 'linear', 'poly'], 
        'C': [0.001, 0.01, 0.1],
        'epsilon': [0.1, 0.15, 0.2], 
        'gamma': [0.001, 0.01, 0.1]
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(svr, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    grid_search.fit(X_train_feats, y_train_tgt)
    
    print(f"Best hyperparameters for {target_name} ({arch_name}): {grid_search.best_params_}")
    best_svr_model = grid_search.best_estimator_
    y_pred = best_svr_model.predict(X_test_feats)
    
    mape = mean_absolute_percentage_error(y_test_tgt, y_pred)
    print(f"Test MAPE for {target_name} ({arch_name}): {mape:.2f}%")
    
    return best_svr_model, mape