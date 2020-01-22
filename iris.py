import sys
print('Python: ()'.format(sys.version))
# scipy
import scipy
print('scipy: ()'.format(scipy.__version__))
# numpy
import numpy
print('numpy: ()'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: ()'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: ()'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: ()'.format(sklearn.__version__))

# libraris
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = pandas.read_csv(url,names=names)
print(dataset.shape)

print(dataset.head(30))

print(dataset.describe())

print(dataset.groupby('class').size())

dataset.plot(kind='box',subplots = True, layout = (2,2),sharex = False,sharey = False)
plt.show()


dataset.hist()
plt.show()



scatter_matrix(dataset)
plt.show

array = dataset.values
x = array[:,0:4]
y = array[:,4]
validation_size = 0.20
seed = 6
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size = validation_size,random_state=seed)



seed = 6
scoring = 'accuracy'
