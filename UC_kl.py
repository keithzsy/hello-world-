import quandl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set()

p4tc = quandl.get("LLOYDS/BPI", authtoken="SzFxrPHJw5zYqRLHtz5J")

resample_method = 'W'

resample_period = 52

resample_harmonics = 10
#, 'harmonics': resample_harmonics
after_14 = p4tc.loc['2015-01-01':].resample(resample_method).mean()/100
print(after_14)

unobserve_model1 = {
            'level': 'fixed intercept',
            #'irregular': True,
            'cycle' : True,
            #'damped_cycle' : True,
            #'freq_seasonal': [{'period': resample_period, 'harmonics': resample_harmonics}],
            #'stochastic_freq_seasonal': [True],
            #'autoregressive' : 1,
            'freq': resample_method
        }

mod_ll = sm.tsa.UnobservedComponents(after_14, **unobserve_model1)
result = mod_ll.fit(method='powell', disp=False)
result.plot_components()
result.plot_diagnostics()
plt.show()