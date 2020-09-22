from sklearn.model_selection import train_test_split


def build_model(x,y,tsize,model_name):
    assert (0. <= tsize <= 1.)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=tsize,random_state=42)

    if model_name =='Random Forest':
        print('You selected Random Forest')

        e_mse, e_rmse, e_mae, e_r2, e_agg = Model.build_rforest(x_train,x_test,y_train,y_test,True)

        print('MSE:', e_mse)
        print('RMSE:', e_rmse)
        print('MAE:', e_mae)
        print('R2:', e_r2)
        print('AGM:', e_agg)

    return

