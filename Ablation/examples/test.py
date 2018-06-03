from Ablation.ablation.main import *

import pandas as pd


def example1():
    name = 'TZN'
    df = pd.read_excel('Threshold Fluence Attempt6.xlsx', sheet_name=name)
    df = df.dropna()
    d = D2LnF(df)
    # d.plot_f_th()
    fit, fig = d.incubation("glass")
    fig.savefig('test')
    plt.show()
    for k, v in fit.items():
        print('{:9} \t {:.2g}'.format(k, v))


def example2():
    name = 'TZN_AFM'
    df = pd.read_excel('Threshold Fluence Attempt6.xlsx', sheet_name=name)
    d = AFM(df)
    d.height()

if __name__ == "__main__":
    # journalplotting()
    example2()
