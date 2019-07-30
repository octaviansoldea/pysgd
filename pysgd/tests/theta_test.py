import pytest
import numpy as np
from pysgd import sgd

def get_data(obj):
    datafile= dict(
        linear='regression.txt',
        logistic='logistic.txt'
    )
    datafromfile = np.loadtxt(open('tests/' + datafile[obj], 'rb'), delimiter=',')
    data = np.ones((datafromfile.shape[0], (datafromfile.shape[1] + 1)))
    data[:,1:] = datafromfile
    mu = np.mean(data[:,1:-1], axis=0)
    sigma = np.std(data[:,1:-1], axis=0)
    data[:,1:-1] = (data[:,1:-1] - mu) / sigma
    return data

@pytest.mark.parametrize('obj', ['stab_tang', 'logistic', 'polynomial']
                         )
@pytest.mark.parametrize('adapt', ['constant', 'adagrad', 'adam']
                         )
def test_theta(obj, adapt):
    if obj == 'stab_tang':
        data = np.array([])
        theta0 = np.array([-0.2, -4.4])
    elif obj == 'polynomial':
        #data = np.array([[-1, 0, 1, 2], [1, 0, 1, 4]])
        data = np.array([[-1.9, -0.9, 0.1, 1.1, 2.1], [4, 1, 0, 1, 4]])
        theta0 = np.array([1.5])
    else:
        data = get_data(obj)
        theta0 = np.zeros(data.shape[1] - 1)

    adapts =      ['constant', 'adagrad', 'adam']
    alpha = dict(
        stab_tang= [   0.01,      0.10,     0.10 ],
        linear=    [   0.01,      0.50,     0.05 ],
        logistic=  [   0.30,      0.30,     0.10 ],
        polynomial=[   0.10,      0.10,     0.10]
    )

    theta_hist, theta = sgd(
        theta0=theta0,
        obj=obj,
        adapt=adapt,
        data=data,
        alpha=alpha[obj][adapts.index(adapt)],
        iters=2000
    )

    result = dict(
        stab_tang = np.array([-2.9,  -2.9]),
        linear = np.array([340412.65957447,  109447.79624389,   -6578.35462]),
        logistic = np.array([1.71617081,  3.98250812,  3.73291532]),
        polynomial=np.array([0.1])
    )

    print(theta_hist[-10])

    if obj == 'logistic':
        assert np.allclose(theta_hist[-1,:-1], result[obj], atol=0.1) #== True
        print("+++++++++++++++++++++++++++++++ theta_hist[-1,:-1] = ", theta_hist[-1,:-1])
    else:
        assert np.allclose(theta, result[obj], atol=0.01)  # == True
        print("+++++++++++++++++++++++++++++++ theta = ", theta)
