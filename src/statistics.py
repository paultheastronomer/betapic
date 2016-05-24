import numpy as np

class Stats:
    '''
    
    '''
    #def __init__(self):
    #    self.Welcome()

    def Welcome(self):
        print "Welcome"
        
    def chi2(self, X):
        '''
        Calculated the Chi2 value
        X[0] => Obs
        X[1] => Err
        X[2] => Model
        '''
        return np.sum(((X[0] - X[2]) / X[1])**2.)
    
    def Merit(self, X):
        ''' Given a Chi2 value
        we calculate a likelihood as our merit function
        '''
        return np.exp(-self.chi2(X)/2.)
