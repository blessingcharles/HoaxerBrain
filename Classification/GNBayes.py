import numpy as np

class GaussianNaiveBayes:
    def __init__(self , verbose : bool = False) -> None:
        self.verbose = verbose
    
    def fit(self, X:np.ndarray , y : np.ndarray):
        """
            Gaussian Naive Bayes
            p(c|features) = p(c)*p(f1/c)*p(f2/c)....p(fn/c)

            p(f(i)/c) ---> probability function in gaussian distribution graph
            which is given by 1/(std*sqrt(2*pi)) * exp[ -1/2*(x-mean/std)^2 ]
            [convert that formula for variance] =>  1/sqrt((var*2*pi)) * exp[ -(x-mean)^2/(2*var) ]

            self._classes            : unique classes in target label
            self._prior_classes_prob : prior probability for class
            self._mean               : mean for each feature in each class ie shape(classes , feature)
            self._var                : standard variance for each feature in each class 
                                                                            ie shape(classes , feature)

        """
        n_samples , n_features = X.shape
        self._classes  =  np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.full((n_classes,n_features),0 , dtype=np.float64)
        self._var = np.full((n_classes,n_features),0 , dtype=np.float64)
        self._prior_classes_prob = np.full(n_classes,0 , dtype=np.float64)

        for c in self._classes:
            class_samples = X[c == y]    # get all samples belonging to that class
            self._mean[c , :] = class_samples.mean(axis=0)
            self._var[c , :] = class_samples.var(axis=0)
            self._prior_classes_prob[c] = len(class_samples)/n_samples

    def predict(self,y_test):
        y_pred = [self._predict(sample) for sample in y_test]
        return np.array(y_pred)

    def _predict(self , sample):
        """
            use gaussian bayes theorem for each classes and find the class which has the maximum probability
            we are taking logarithmic to avoid underflow ie multiplication very small fraction can lead to
        
        """
        self.classes_prob = []

        for c in self._classes:
            prior_probability = np.log(self._prior_classes_prob[c])
            summation_all_gd = np.sum(np.log((self._gaussion_distribution(c,sample))))
            class_prob = prior_probability+summation_all_gd

            self.classes_prob.append(class_prob)

        return self._classes[np.argmax(self.classes_prob)]

    def _gaussion_distribution(self,c,x):
        mean = self._mean[c]
        variance = self._var[c]

        numerator = np.exp(-(x-mean)**2 / (2*variance))
        denominator = np.sqrt(2*np.pi*variance)

        return numerator/denominator

