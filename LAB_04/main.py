from lab4 import *

root = r"Dataset"
X, y = build_dataset(root)

#we only have to use 2 classes
mask = (y == 0) | (y == 1)
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #70-30 split

model = KNeighborsClassifier(n_neighbors=3) #initialise with k=3
model.fit(X_train, y_train) #train 

y_pred_train = model.predict(X_train) #predict labels with training data
y_pred_test = model.predict(X_test) #now testing data

#evaluate classification on training data and testing data
cm_train, p_tr, r_tr, f_tr = evaluate_classification(y_train, y_pred_train)
cm_test, p_te, r_te, f_te = evaluate_classification(y_test, y_pred_test)

    #A1
print("Train CM:\n", cm_train)
print("Test CM:\n", cm_test)
print("Test Precision:", p_te)
print("Test Recall:", r_te)
print("Test F1:", f_te)
#overfit model

    #A2
#calculate mse rmse mape and r2 score
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
X_reg = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
y_reg = df["Payment (Rs)"].values
X_reg_aug = np.c_[np.ones(X_reg.shape[0]), X_reg]
weights = np.linalg.lstsq(X_reg_aug, y_reg, rcond=None)[0]
y_pred_reg = np.dot(X_reg_aug, weights)
mse, rmse, mape, r2 = regression_metrics(y_reg, y_pred_reg)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)
print("R2:", r2)

    #A3
X_syn, y_syn = generate_training_points()

plt.scatter(X_syn[:,0], X_syn[:,1], c=y_syn)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("A3 – Synthetic Training Data")
plt.show()

    #A4 and A5
grid = generate_test_grid()
for k in [1, 3, 5]:
    preds = classify_grid(X_syn, y_syn, grid, k)
    plt.scatter(grid[:,0], grid[:,1], c=preds, s=1)
    plt.title(f"A4 & A5 – k={k}")
    plt.show()

    #A6
X_proj2, y_proj2 = select_two_features_two_classes(X, y)
Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(X_proj2, y_proj2, test_size=0.3, random_state=42)
model.fit(Xp_tr, yp_tr)

y_pred_a6 = model.predict(Xp_te)
acc_a6 = model.score(Xp_te, yp_te)

print("Accuracy:", acc_a6)

plt.scatter(Xp_te[:,0], Xp_te[:,1], c=y_pred_a6)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("A6")
plt.show()


    #A7
#hyper parameter tuning 
best_k = tune_k(X_train, y_train)
print(best_k)