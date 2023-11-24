import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import yaml 


def groupes_politiques(row_classes, deputes): 
    dep = pd.DataFrame(deputes)['groupe']
    groups = dep.unique()
    count  = np.zeros_like(groups)
    group_per_class = []
    df_counts = pd.DataFrame(columns=pd.Index(groups))

    for nq in range(len(deputes)): 
        vec = []
        for i, c in enumerate(row_classes): 
            if c==nq: 
                count[np.where(dep[i]==groups)[0]] += 1
                if dep[i] not in vec: 
                    vec.append(dep[i])
        if vec != []: 
            group_per_class.append(vec)
            df_counts.loc[len(df_counts)+1] = pd.Series(count, index=pd.Index(groups))
            
    return group_per_class,df_counts 

def pi_df(pi, row_classes, column_classes): 
    pi_ = np.zeros((len(row_classes), len(column_classes)))
    for i,t1 in enumerate(row_classes): 
        for j,t2 in enumerate(column_classes): 
            pi_[i,j] = pi[t1, t2]
    return pi_

def text_legend_row(gpp, df_counts):
    gp = gpp.copy()
    for i, gp_ in enumerate(gp): 
        c = df_counts.iloc[i]
        for j, parti in enumerate(gp_): 
            gp[i][j] = str(gp_[j])+ ' ('+ str(c[parti]) + ')'
    return gp 

def generate_alphabet_text_array(n):
    """
    Generate a text array with letters from 'A' to the nth letter.

    Parameters:
    - n (int): Number of letters to include.

    Returns:
    - text_array (numpy.ndarray): Numpy array of strings containing the letters.
    """
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letters = np.array(list(alphabet)[:n])
    return letters

def fig_17(votes, deputes, row_classes, column_classes, pi): 
    nl = len(np.unique(column_classes))
    nq = len(np.unique(row_classes))

    #Create the matrix with probability clusters 
    pi_ = pi_df(pi, row_classes, column_classes)

    #Political group clusters and the count per group
    gp, df_counts = groupes_politiques(row_classes, deputes)

    #defines political groups in each row cluster and number of members from each group 
    r_legend = text_legend_row(gp, df_counts)
    m_text = '\n\n\n'.join(['\n'.join(row) for row in r_legend])

    c_legend = (' '*40).join([(' '*40).join(row) for row in generate_alphabet_text_array(nl)])

    ### Figure creation 
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize= (20,6))

    ax0 = ax[0].imshow(votes[np.argsort(row_classes),:][:,np.argsort(column_classes)], cmap='gray', label='ordered', )
    ax[0].contour(pi_[np.argsort(row_classes),:][:,np.argsort(column_classes)], levels=np.unique(pi_), colors='red', linewidths=0.5)
    ax[0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,labelbottom=False, labelleft=False)
    ax[0].text(-200, 525, m_text, fontsize=7,  wrap=True)  # Add text in the left subplot
    ax[0].text(50, 650, c_legend, fontsize=10,  wrap=True)  # Add text in the left subplot

    cbar0 =plt.colorbar(ax0, ax = ax[0],location='bottom', ticks= np.unique(votes), shrink=0.6)
    cbar0.set_ticklabels(['Negative','NA','Positive'])
    cbar0.set_label('Votin result' )


    ax1 = ax[1].imshow(pi_[np.argsort(row_classes),:][:,np.argsort(column_classes)], cmap='gray', label='clusters')
    ax[1].text(50, 650, c_legend, fontsize=10,  wrap=True)  # Add text in the left subplot
    ax[1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,labelbottom=False, labelleft=False)
    cbar1 = plt.colorbar(ax1, shrink=0.6, ax = ax[1], location='bottom')
    cbar1.set_label('Proba of voting positively')

    plt.tight_layout()
    plt.savefig('Figures/17.png')

    plt.show()


def fig_nu(nu_i, nu_j, dfr, string_char):
    data = {
        string_char[0]: np.array([dfr[i][string_char[0]] for i in range(len(dfr))]),
        string_char[1] : nu_i.reshape(-1),
        string_char[2] :  nu_j.reshape(-1)}
    dfr = pd.DataFrame(data)
    plt.figure()
    sns.scatterplot(x=string_char[1],y=string_char[2], data=dfr, hue=string_char[0]).set(title="Maximum a posteriori estimates of the MPs")
    plt.xlabel(string_char[1])
    plt.ylabel(string_char[2])
    plt.legend(fontsize='8') # for legend text
    plt.savefig('Figures/'+string_char[1]+'_'+string_char[2]+'.png')
    pass