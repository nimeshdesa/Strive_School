
class linearregression():
    def __init__(self): ## Class initialization
        self.n_value = 0
        self.a = 0
        self.b = 0
        self.score = 0
        self.x_mean = 0
        self.y_mean = 0
        self.rsme = 0
        
    def fit(self,X,Y): ## Fit method

        self.a, self.b, self.x_mean, self.y_mean = self.leastSquare(X,Y)
        self.r2(X,Y)
        self.Rsme(X,Y)
        
    def leastSquare(self,X,Y): 

        x_mean = np.mean(X)
        y_mean = np.mean(Y)

        self.n_values = len(X)

        numerator=0
        denomrator=0 

        for i in range(self.n_values):
            numerator += (X[i] - x_mean) * (Y[i] - y_mean)
            denomrator += np.square(X[i] - x_mean)

        a = numerator / denomrator
        b = y_mean - (a * x_mean)

        return a,b, x_mean, y_mean

    def r2(self, X,Y):
        sst = 0
        ssr = 0
        for i in range(self.n_values):
            y_pred = self.b + self.a * X[i]
            sst += np.square(Y[i] - self.y_mean)
            ssr += np.square(Y[i] - y_pred)
        
        self.score = 1 - (ssr/sst)
        
    def Rsme(self, X, Y):
        
        rsme=0
        for i in range(self.n_values):
            y_pred = self.b + self.a * X[i]
            rsme = np.square(Y[i] - y_pred)
        
        self.rsme = np.sqrt(rsme/self.n_values)