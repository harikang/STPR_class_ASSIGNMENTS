import numpy as np
import matplotlib.pyplot as plt
# Define means and covariances
mu1 = np.array([2, 2])
sigma1 = np.array([[5, 0], [0, 1]])
mu2 = np.array([2, 5])
sigma2 = np.array([[1, 0], [0, 3]])
# Set random seed for reproducibility
np.random.seed(88)
# Generate data
r1 = np.random.multivariate_normal(mu1, sigma1, size=1000)
r2 = np.random.multivariate_normal(mu2, sigma2, size=1000)
X = np.concatenate((r1, r2), axis=0)
# Separate data for training and testing (mimicking the approach)
X_train = np.concatenate((X[:200], X[1000:1200]))
X_test = np.concatenate((X[200:1000], X[1201:2000]))
y_train = np.concatenate((-np.ones(200), np.ones(200)))
y_test = np.concatenate((-np.ones(800), np.ones(800)))
# Create the scatter plot
plt.figure(1)
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.7) # Scatter plot
plt.plot(X_train[:, 0], X_train[:, 1], ’b+’, label=’Training Set - Class 1’)
plt.plot(X_train[200:, 0], X_train[200:, 1], ’ro’, label=’Training Set - Class 2’)
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], s=10, alpha=0.7) # Scatter plot
plt.plot(X_test[:, 0], X_test[:, 1], ’b+’, label=’Test Set - Class 1’)
plt.plot(X_test[800:, 0], X_test[800:, 1], ’ro’, label=’Test Set - Class 2’)
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(77)
# Parameters for the Gaussian distribution
# Mean and covariance for the training set
mean_data1 = [0, 0]
cov_data1 = [[3, 0], [0, 5]]  # Diagonal covariance, for independence

# Mean and covariance for the test set
mean_data2 = [2, 2]
cov_data2 = [[1, 0], [0, 1]]  # Diagonal covariance, for independence
# Generating samples
data1= np.random.multivariate_normal(mean_data1, cov_data1, 1000)
data2 = np.random.multivariate_normal(mean_data2, cov_data2, 1000)
classm1 = -np.ones((1000, 1))  # Class -1
class1 = np.ones((1000, 1))  # Class 1
# Separate data for training and testing (mimicking the approach)
X_train = np.concatenate((data1[:200], data2[:200]))
X_test = np.concatenate((data1[200:], data2[200:]))
y_train = np.concatenate((classm1[:200], class1[:200]))
y_test = np.concatenate((classm1[200:], class1[200:]))
# Plotting
plt.figure(figsize=(10, 5))

# Training data
plt.subplot(1, 2, 1)
plt.scatter(X_train[:200, 0], X_train[:200, 1], c='blue', label='Class -1', alpha=0.5)
plt.scatter(X_train[200:, 0], X_train[200:, 1], c='red', label='Class 1', alpha=0.5)
plt.title('Training Data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()

# Testing data
plt.subplot(1, 2, 2)
plt.scatter(X_test[:800, 0], X_test[:800, 1], c='blue', label='Class -1', alpha=0.5)
plt.scatter(X_test[800:, 0], X_test[800:, 1], c='red', label='Class 1', alpha=0.5)
plt.title('Testing Data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()

plt.tight_layout()
plt.show()

def LR(X,y):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    return  np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)   

# loss function 
def RSS(y_pred,y):
    return np.sum((y - y_pred) ** 2) 

w = LR(X_train,y_train)
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_test_pred = X_test_bias @ w
ktest = RSS(y_test_pred,y_test)
print(ktest)

X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
y_pred = X_train_bias @ w 
ktrain = RSS(y_pred,y_train)
print(ktrain)

# draw decision boundary 
x_values = np.array([np.min(X_train[:, 0]), np.max(X_train[:, 0])])
y_values = -(w[1] / w[2]) * x_values - (w[0] / w[2])
plt.figure(figsize=(10, 5))

# Training data
plt.subplot(1, 2, 1)
plt.scatter(X_train[:200, 0], X_train[:200, 1], c='blue', label='Class -1', alpha=0.5)
plt.scatter(X_train[200:, 0], X_train[200:, 1], c='red', label='Class 1', alpha=0.5)
plt.plot(x_values, y_values, label="Decision Boundary")
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Boundary in Train Dataset')
plt.legend()

# Testing data
plt.subplot(1, 2, 2)
plt.scatter(X_test[:800, 0], X_test[:800, 1], c='blue', label='Class -1', alpha=0.5)
plt.scatter(X_test[800:, 0], X_test[800:, 1], c='red', label='Class 1', alpha=0.5)
plt.plot(x_values, y_values, label="Decision Boundary")
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Boundary in Test Dataset')
plt.legend()


plt.show()

def RM(X, order):
    # Build regressor matrix P (mxK):
    # order = desired order of approximation,
    # X = input matrix (mxl), K = number of parameters to be est.
    # m = number of data samples, l = input dimension.
    m, l = X.shape
    MM1 = []
    MM3 = []
    Msum = np.sum(X, axis=1)
    for i in range(1, order+1):
        M1 = np.zeros((m, l))
        M3 = np.zeros((m, l))
        for k in range(l):
            M1[:, k] = X[:, k]**i
            if i > 1:
                M3[:, k] = X[:, k] * Msum**(i-1)
        MM1.append(M1)
        if i > 1:
            MM3.append(M3)
    if MM3:
        P = np.concatenate([np.ones((m, 1)), np.concatenate(MM1, axis=1), np.concatenate(MM3, axis=1)], axis=1)
    else:
        P = np.concatenate([np.ones((m, 1)), np.concatenate(MM1, axis=1)], axis=1)
    return P

X = np.array([[1, 2], [3, 4], [5, 6]])
order = 2
P = RM(X, order)
print(P)

P_train = {}
for i in range(1,11):
    P_train[f'P_train{i}'] = RM(X_train,i)

def PLR(P,y):
    return np.linalg.inv(P.T @ P) @ P.T @ y   

LR_weights = {}
for i in range(1,11):
    train = P_train[f'P_train{i}']
    LR_weights[f'LR_train{i}'] = PLR(train,y_train)    
    print(f'{i}th order RM regression weight shape is ',LR_weights[f'LR_train{i}'].shape)

predicts = {}
P_test = {}
for i in range(1,11):
    P_test[f'P_test{i}'] = RM(X_test,i)
    test = P_test[f'P_test{i}']
    weight = LR_weights[f'LR_train{i}']
    predicts[f'order{i}_pred'] = test @ weight 

# i=0
# for k,v in predicts.items():
#     i+=1
#     print(f"{i}th order RM model")
#     print(f"{v[:10]}")

# # draw decision boundary 
# w = LR_weights['LR_train5']
# x_values = np.array([np.min(X_train[:, 0]), np.max(X_train[:, 0])])
# y_values = -(w[1] / w[2]) * x_values - (w[0] / w[2])
# plt.figure(figsize=(10, 5))

# # Training data
# plt.subplot(1, 2, 1)
# plt.scatter(X_train[:200, 0], X_train[:200, 1], c='blue', label='Class -1', alpha=0.5)
# plt.scatter(X_train[200:, 0], X_train[200:, 1], c='red', label='Class 1', alpha=0.5)
# plt.plot(x_values, y_values, label="Decision Boundary")
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.title('Decision Boundary in Train Dataset')
# plt.legend()

# # Testing data
# plt.subplot(1, 2, 2)
# plt.scatter(X_test[:800, 0], X_test[:800, 1], c='blue', label='Class -1', alpha=0.5)
# plt.scatter(X_test[800:, 0], X_test[800:, 1], c='red', label='Class 1', alpha=0.5)
# plt.plot(x_values, y_values, label="Decision Boundary")
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.title('Decision Boundary in Test Dataset')
# plt.legend()


# plt.show()
# plt.figure(figsize=(15, 10))

# # Loop through orders 1 to 6
# for i in range(1, 7):
#     w = LR_weights[f'LR_train{i}']
#     x_values = np.array([np.min(X_train[:, 0]), np.max(X_train[:, 0])])
#     y_values = -(w[1] / w[2]) * x_values - (w[0] / w[2])
    
#     # Training data plot
#     plt.subplot(2, 3, i)
#     plt.scatter(X_train[:200, 0], X_train[:200, 1], c='blue', label='Class -1', alpha=0.5)
#     plt.scatter(X_train[200:, 0], X_train[200:, 1], c='red', label='Class 1', alpha=0.5)
#     plt.plot(x_values, y_values, label=f"Decision Boundary (Order {i})")
#     plt.xlabel('X1')
#     plt.ylabel('X2')
#     plt.title(f'Decision Boundary for Order {i}')
#     plt.legend()

# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(15, 10))

# # Loop through orders 1 to 6
# for i in range(1, 7):
#     w = LR_weights[f'LR_train{i}']
#     x_values = np.array([np.min(X_test[:, 0]), np.max(X_test[:, 0])])
#     y_values = -(w[1] / w[2]) * x_values - (w[0] / w[2])
    
#     # testing data plot
#     plt.subplot(2, 3, i)
#     plt.scatter(X_test[:800, 0], X_test[:800, 1], c='blue', label='Class -1', alpha=0.5)
#     plt.scatter(X_test[800:, 0], X_test[800:, 1], c='red', label='Class 1', alpha=0.5)
#     plt.plot(x_values, y_values, label=f"Decision Boundary (Order {i})")
#     plt.xlabel('X1')
#     plt.ylabel('X2')
#     plt.title(f'Decision Boundary for Order {i}')
#     plt.legend()

# plt.tight_layout()
# plt.show()

predicts = {}
P_test = {}
for i in range(1,7):
    P_test[f'P_test{i}'] = RM(X_test,i)
    test = P_test[f'P_test{i}']
    weight = LR_weights[f'LR_train{i}']
    predicts[f'order{i}_pred'] = test @ weight 

def classacc(ytest,ypred):
    y_pred_class = np.where(ypred >= 0, 1, -1)
    correct_classifications = y_test == y_pred_class
    accuracy = np.mean(correct_classifications)
    return accuracy
i=0
for v in predicts.values():
    acc = classacc(y_test,v)
    i+=1
    print(f'RM {i}th order RM Model acc is ', f'{acc*100:.2f}','%')
def LR(X,y):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    return  np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)    
w = LR(X_train,y_train)
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_test_pred = X_test_bias @ w

acc = classacc(y_test,y_test_pred)
print(f'LR Model acc is ', f'{acc*100:.2f}','%')
# Data for RM model orders and their accuracies
orders = list(range(1, 11))
accuracies = [0.84, 0.73, 0.82, 0.81, 0.85, 0.56, 0.76, 0.62, 0.48, 0.52]

# Plotting the accuracies
plt.figure(figsize=[10,6])
plt.plot(orders, accuracies, marker='o', linestyle='-', color='b')
plt.title('Test Classification Accuracies of the RM Model')
plt.xlabel('Order of RM Model')
plt.ylabel('Accuracy (%)')
plt.xticks(orders)
plt.grid(True)

for i, txt in enumerate(accuracies):
    plt.text(orders[i], accuracies[i], f"{txt*100:.1f}%", ha='center', va='bottom')
plt.show()
