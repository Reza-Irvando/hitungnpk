mae = mean_absolute_error(y_test_tgt, y_pred)
mse = mean_squared_error(y_test_tgt, y_pred)
rmse = np.sqrt(mse)

print(f"Test MAE  : {mae:.4f}")
print(f"Test MSE  : {mse:.4f}")
print(f"Test RMSE : {rmse:.4f}")