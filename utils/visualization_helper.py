import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.decomposition import PCA
def pom_frame(pom_embeds, y, dir, required_desc, title, size1, size2, size3):
    sns.set_style("ticks")
    sns.despine()

    # pom_embeds = model.predict_embedding(dataset)
    # y_preds = model.predict(dataset)
    # required_desc = list(dataset.tasks)
    type1 = {'floral': '#F3F1F7', 'subs': {'muguet': '#FAD7E6', 'lavender': '#8883BE', 'jasmin': '#BD81B7'}}
    type2 = {'meaty': '#F5EBE8', 'subs': {'savory': '#FBB360', 'beefy': '#7B382A', 'roasted': '#F7A69E'}}
    type3 = {'ethereal': '#F2F6EC', 'subs': {'cognac': '#BCE2D2', 'fermented': '#79944F', 'alcoholic': '#C2DA8F'}}

    # Assuming you have your features in the 'features' array
    pca = PCA(n_components=2,
              iterated_power=10)  # You can choose the number of components you want (e.g., 2 for 2D visualization)
    reduced_features = pca.fit_transform(pom_embeds)  # try different variations

    variance_explained = pca.explained_variance_ratio_

    # Variance explained by PC1 and PC2
    variance_pc1 = variance_explained[0]
    variance_pc2 = variance_explained[1]

    # if is_preds:
    #     y = np.where(y_preds>threshold, 1.0, 0.0) # try quartile range (or rank)
    # else:
    #     y = dataset.y

    # Generate grid points to evaluate the KDE on (try kernel convolution)
    x_grid, y_grid = np.meshgrid(np.linspace(reduced_features[:, 0].min(), reduced_features[:, 0].max(), 500),
                                 np.linspace(reduced_features[:, 1].min(), reduced_features[:, 1].max(), 500))
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])

    def get_kde_values(label):
        plot_idx = required_desc.index(label)
        # print(y[:, plot_idx])
        label_indices = np.where(y[:, plot_idx] == 1)[0]
        kde_label = gaussian_kde(reduced_features[label_indices].T)
        kde_values_label = kde_label(grid_points)
        kde_values_label = kde_values_label.reshape(x_grid.shape)
        return kde_values_label

    def plot_contours(type_dictionary, bbox_to_anchor):
        main_label = list(type_dictionary.keys())[0]
        plt.contourf(x_grid, y_grid, get_kde_values(main_label), levels=1,
                     colors=['#00000000', type_dictionary[main_label], type_dictionary[main_label]])
        axes = plt.gca()  # Getting the current axis

        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        legend_elements = []
        for label, color in type_dictionary['subs'].items():
            plt.contour(x_grid, y_grid, get_kde_values(label), levels=1, colors=color, linewidths=2)
            legend_elements.append(Patch(facecolor=color, label=label))
        legend = plt.legend(handles=legend_elements, title=main_label, bbox_to_anchor=bbox_to_anchor, prop={'size': 30})
        legend.get_frame().set_facecolor(type_dictionary[main_label])
        plt.gca().add_artist(legend)

    fig = plt.figure(figsize=(15, 15), dpi=700)
    # ax.spines[['right', 'top']].set_visible(False)
    # plt.title('KDE Density Estimation with Contours in Reduced Space')
    # plt.xlabel(f'Principal Component 1 ({round(variance_pc1*100, ndigits=2)}%)')
    # plt.ylabel(f'Principal Component 2 ({round(variance_pc2*100, ndigits=2)}%)')
    plt.xlabel('Principal Component 1', fontsize=35)
    plt.ylabel('Principal Component 2', fontsize=35)
    plot_contours(type_dictionary=type1, bbox_to_anchor=size1)
    plot_contours(type_dictionary=type2, bbox_to_anchor=size2)
    plot_contours(type_dictionary=type3, bbox_to_anchor=size3)
    # plt.colorbar(label='Density')
    # plt.show()
    # png_file = os.path.join(dir, 'pom_frame.png')
    # plt.savefig(png_file)
    plt.savefig("figs/realign_islands" + title + ".svg")
    plt.savefig("figs/realign_islands" + title + ".pdf")
    plt.show()
    plt.close()

def plot_lines(data, title, filename):
    df_corrs = pd.DataFrame.from_dict(data, orient='index',
                                      columns=['Dataset', 'Correlation'])
    df_corrs = df_corrs.explode('Correlation')
    df_corrs['Layer'] = df_corrs.groupby(level=0).cumcount()
    # alternative
    # df['idx'] = df.groupby(df.index).cumcount()
    df_corrs = df_corrs.reset_index(drop=True)
    sns.set_style("white")
    # df[['0', '1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11']] = df['All layers'].str.split(' ', expand=True)
    fig, ax = plt.subplots(figsize=(10, 10)
                           # ,constrained_layout = True
                           )

    # sns.color_palette("tab10")
    sns.color_palette("hls", 4)
    palette = ['#4d79a4', '#ecc947', '#b07aa0']
    g = sns.lineplot(data=df_corrs, x="Layer", y="Correlation", hue="Dataset", palette=palette, lw=7)
    # sns.barplot(df_corrs, x="Dataset", y="Correlation", hue= "Correlation",width=0.2,legend=False,palette=sns.color_palette("Set2",4))

    ax.legend().set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    # axes = plt.gca() #Getting the current axis

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 56c4c0fb

    fig.subplots_adjust(bottom=0.35, left=0.2)
    fig.legend(handles, labels, ncol=3, columnspacing=1, prop={'size': 25}, handlelength=1.5, loc="lower center",
               borderpad=0.4,

               bbox_to_anchor=(0.54, 0.07),

               frameon=True, labelspacing=0.4, handletextpad=0.2)
    g.set_xticks([0, 2, 4, 6, 8, 10])  # <--- set the ticks first
    g.set_xticklabels(['1', '3', '5', '7', '9', '11'])

    # g.set_yticks([0.45,0.5,0.55,0.6,0.65,0.7]) # <--- set the ticks first
    # g.set_yticklabels(['', '0.5','', '0.6','', '0.7'])

    # g.set_yticks([0.45,0.5,0.55,0.6,0.65,0.7]) # <--- set the ticks first
    # g.set_yticklabels(['', '0.5','', '0.6','', '0.7'])
    # g.set_ylim(0.45,0.7)

    g.set_xlim(0, 11)
    ax.set_ylabel('')
    ax.set_xlabel('Model Layer')
    # plt.margins(0,-0.16)
    ax.xaxis.set_label_coords(0.5, -0.18)

    # plt.tight_layout()
    plt.savefig(filename
                , bbox_inches="tight"

                )


def plot_bars(data, title, filename):
    df_corrs = pd.DataFrame.from_dict(data, orient='index',
                                      columns=['Dataset', 'type', 'Correlation'])
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 8)
                           # ,constrained_layout = True
                           )
    # sns.set_style("whitegrid")
    # sns.color_palette("tab10")
    sns.color_palette("hls", 4)
    palette = ["#91bfdb",
               "#003f5c",
               # "#d73027",
               # "#fc8d59",
               # "#ffda33",

               # , "#4575b4"
               ]

    palette = ['#4d79a4', '#ecc947', '#b07aa0']

    # plt.subplots_adjust(bottom=0.3)
    g = sns.barplot(df_corrs, x="Dataset", y="Correlation", hue="type", width=0.4, palette=palette)
    # for i in ax.containers:
    #     ax.bar_label(i,fmt="%2.2f")
    ax.legend().set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.subplots_adjust(bottom=0.35, left=0.2)
    fig.legend(handles, labels, ncol=3, columnspacing=0.8, prop={'size': 25}, handlelength=1.5, loc="lower center",
               borderpad=0.4,

               bbox_to_anchor=(0.54, 0.07),

               frameon=True, labelspacing=0.4, handletextpad=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # g.set_yticks([0.45,0.5,0.55,0.6,0.65,0.7]) # <--- set the ticks first
    # g.set_yticklabels(['', '','', '','', ''])

    ax.set_ylabel('Correlation Coefficient')

    # plt.margins(x=0.5)
    # fig.update_layout(bargap=0.1)
    # ax.set_subtitle("Dataset")  # Set title with numeric distance
    # axes[0].set_title("Total Bill Distribution", loc=(0.05, 0.9))  # Set title with numeric distance

    # ax.set_title('')
    ax.xaxis.set_label_coords(0.5, -0.18)
    # plt.tight_layout()
    #     g.set_yticks([0.45,0.5,0.55,0.6,0.65,0.7]) # <--- set the ticks first

    #     g.set_yticklabels(['', '0.5','', '0.6','', '0.7'])
    #     g.set_ylim(0.45,0.7)

    plt.savefig(filename,
                bbox_inches="tight"
                )