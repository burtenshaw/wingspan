#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = '/home/burtenshaw/now/spans_toxic/figures/performace_'
# %% SUB WORD FEATURES


def make_plot(models, bars=None, dev_data = [0,1], title='', y2label = 'Mean F1 Score', y1label = '', xlabel='Number of toxic words', ticks=[]):

    mdf = pd.DataFrame(models)
    _shape = mdf.shape[0]
    mdf['id'] = mdf.index
    mdf = mdf.melt(id_vars=['id'])
    mdf['sdev'] = np.tile(np.arange(0.05, 1, 1/_shape), int(mdf.shape[0]/_shape))

    mdf['upper'] = mdf.value + (mdf.sdev/2)
    mdf['lower'] = mdf.upper - mdf.sdev
    mdf = mdf.pivot(columns = 'variable', values=['value', 'upper', 'lower'], index='id')
    
    if ticks == []:
        x_val = ticks = np.arange(0, len(mdf.index), step=1)
        bw = .2
    else:
        x_val = ticks
        bw = 2

    mdf['x'] = x_val

    pal = sns.color_palette("pastel", 4)
    fig, ax1 = plt.subplots(dpi=300)
    ax2 = ax1.twinx()
    ax1.set_ylabel(y1label)
    ax1.set_xlabel(xlabel)
    
    if bars:
        bdf = pd.DataFrame(bars).T
        bdf.columns = ['dev', 'test']
        bdf['x'] = x_val
        bdf = bdf.melt(id_vars = 'x')
        e = sns.color_palette("Blues")[-1]
        f = sns.color_palette("Blues")[1]
        ax2 = sns.barplot(x='x', y="value", hue="variable", data=bdf)
        # ax2.bar(x_val,b, align='edge', color=f, edgecolor=e, width = bw)
        ax2.set_ylabel('Density')
        ax2.set_xlabel(xlabel)

    for n, model in enumerate(models.keys()):
        ls = '--' if 'BASELINE' in model else '-'
        c = '#D15514' if 'test' in model else '#528CC7'
        _x = mdf.x - x_val[0]
        ax1 = sns.lineplot(x=_x, 
                            y=mdf['value'][model], 
                            linewidth=1,
                            color=c, 
                            label=model,
                            linestyle=ls)

        if 'BASELINE' not in model:
            ax1.fill_between(_x, 
                        mdf.lower[model], 
                        mdf.upper[model], 
                        alpha=.1)
    ax1.set_ylim(0,1)
    ax2.set_ylabel(y2label)
    ax1.legend(loc='upper right')
    plt.title(title)
    # plt.xticks(ticks)
    
    return fig


models = {
    'ENSEMBLE_dev' : [0.5188605477831763,
                  0.2776046126083359,
                  0.16137768033809335,
                  0.12818640283006774,
                  0.10369951307277415,
                  0.07131801739117287,
                  0.06027690869801599,
                  0.018072289156626505,
                  0,
                  0],
    'BASELINE_dev' : [0.38406681496072637,
                    0.1812751951017494,
                    0.14364808547552835,
                    0.09343753016266051,
                    0.06673213292398926,
                    0.0610283478044499,
                    0.17837911621100652,
                    0.03824519981945071,
                    0,
                    0],
    'ENSEMBLE_test' : [0.8150706691809341,
                0.5835564094470098,
                0.4947402260344491,
                0.3828831824690896,
                0.2572864940446738,
                .24,
                0.2267605633802817,
                .22,
                0.2440087145969499,
                0],
    'BASELINE_test' : [0.7323803038625091,
                0.5582559167932877,
                0.44769659714583526,
                0.35648141389071386,
                0.4093306341599582,
                0.3,
                0.2,
                .1,
                0.2440087145969499,
                0]
}

bars = [
        [0.46757901596611273,
        0.20886282176604756,
        0.10850439882697947,
        0.07038123167155426,
        0.0433365917236885,
        0.0267188009123493,
        0.02541544477028348,
        0.020202020202020204,
        0.01401107852720756,
        0.014988595633756924],
            [0.6776859504132231,
        0.14049586776859505,
        0.08539944903581267,
        0.04132231404958678,
        0.027548209366391185,
        0.011019283746556474,
        0.008264462809917356,
        0.0027548209366391185,
        0.005509641873278237]
]

fig = make_plot(models, 
                bars, 
                title = 'Model Performance on Toxic Span Length',
                xlabel = 'n Tokens per Toxic Span',
                y1label = 'n Token Frequency',
                ticks = list(np.arange(1,11))
                )
plt.savefig("figures/performance_span_len.png", figsize=(10, 6), dpi=300)
# %% SENTENCE LENGTH

models = {
    'ENSEMBLE_dev' : 
                [0.749967903199466,
                0.6638668568494656,
                0.5111542075534352,
                0.40025233559911944,
                0.2716495009760869,
                0.2630720199414967,
                0.2618803227651217,
                0.25297846365777976,
                0.22203819138703498,
                0.22076393013513987],
    'BASELINE_dev' : 
                [0.6548018682873495,
                0.5331071034218796,
                0.37172253657237586,
                0.2817083291617774,
                0.18375981912333242,
                0.1970855930125419,
                0.17155874012808034,
                0.2358918634391907,
                0.20922931298052372,
                0.18082390949574456],
    'ENSEMBLE_test' : 
                [0.8470248896030141,
                0.7769339131585197,
                0.5404709637366828,
                0.5626086956521739,
                0.4521026870442513,
                0.2535211267605634,
                0.2,
                0.13259762308998302,
                0.11764705882352941,
                0.37037037037037035],
    'BASELINE_test' : [0.774447165040873,
                        0.6766936683342664,
                        0.49022285844888186,
                        0.4751219512195122,
                        0.18579234972677594,
                        0.0,
                        0.0,
                        0.219815668202765,
                        0.11764705882352941,
                        0.37037037037037035]
}

bars = [
        
            [0.6444772513756543,
            0.21594416856797746,
            0.057173533753858544,
            0.026842034626224667,
            0.01328680713998121,
            0.007515769695342907,
            0.004697356059589317,
            0.004697356059589317,
            0.00322104415514696,
            0.0021473627700979736],
        [0.7845579078455791,
        0.18181818181818182,
        0.023661270236612703,
        0.00311332503113325,
        0.0018679950186799503,
        0.0006226650062266501,
        0.0006226650062266501,
        0.0012453300124533001,
        0.0006226650062266501,
        0.0006226650062266501]
]

fig = make_plot(models,
                bars, 
                title='Model Performance on Stop Word Frequency',
                xlabel='n Stop Words per Toxic Span',
                y1label='n Stop Word Frequency',
                ticks = list(np.arange(0,10))
                )
plt.savefig("figures/performance_stop_words.png", figsize=(10, 6), dpi=300)
    # %%

# %%

