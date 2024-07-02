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

    def change_width(ax, new_value):
        for patch in ax.patches:
            current_width = patch.get_width()
            diff = current_width - new_value

            # we change the bar width
            patch.set_width(new_value)

            # we recenter the bar
            patch.set_x(patch.get_x() + diff * .5)

def combine_visualize(df1, df2, df3, tasks, ax, title, type="corr", figure_name="def"):
    df12 = pd.concat((df1, df2))
    df_combined = pd.concat((df12, df3))
    melted_df_keller = df_combined.melt(id_vars=['model'], var_name='descritpor')
    # g1.despine(left=True)
    # g1.set_axis_labels("", title)
    # g1.legend.set_title("")
    # g1.set_xticklabels(tasks, rotation=45)
    if type == "corr":
        melted_df_keller['value'] = melted_df_keller['value'].abs()
    else:
        # melted_df_keller['value'] = melted_df_keller[['value']].apply(np.sqrt)
        pass
    print(melted_df_keller.groupby('model')['value'].mean().reset_index())
    print(melted_df_keller.groupby('model')['value'].sem().reset_index() * 2)
    g1 = sns.barplot(
        data=melted_df_keller,
        x="descritpor", y="value", hue="model",
        errorbar="se", ax=ax, palette=['#4d79a4', '#ecc947', '#b07aa0'], linewidth=7)
    g1.set(xlabel='Model', ylabel=title)
    g1.spines['top'].set_visible(False)
    g1.spines['right'].set_visible(False)
    # g2 = sns.barplot(
    # data=melted_df_keller,
    # x="model", y="value",
    # errorbar="sd", palette="dark", alpha=.6)
    # g2.despine(left=True)
    # g2.set_axis_labels("", "Body mass (g)")
    # g2.legend.set_title("")
    g1.set_xticklabels(tasks, rotation=90)
    # change_width(g1, 0.1)
    # g1.figure.savefig(figure_name+".pdf")
    return g1

def combine_visualize_separate(df1, df2, df3, tasks, ax, title, type="corr", figure_name="def"):
    df12 = pd.concat((df1, df2))
    df_combined = pd.concat((df12, df3))
    melted_df_keller = df_combined.melt(id_vars=['model'], var_name='descritpor')
    # g1 = sns.catplot(
    # data=melted_df_keller, kind="bar",
    # x="descritpor", y="value", hue="model",
    # errorbar="sd", palette="dark", alpha=.6, height=6,aspect =2 )
    # g1.despine(left=True)
    # g1.set_axis_labels("", "Body mass (g)")
    # g1.legend.set_title("")
    # g1.set_xticklabels(tasks, rotation=45)
    if type == "corr":
        melted_df_keller['value'] = melted_df_keller['value'].abs()
    else:
        pass
        # melted_df_keller['value'] = melted_df_keller[['value']].apply(np.sqrt)
    print(melted_df_keller.groupby('model')['value'].mean().reset_index())
    print(melted_df_keller.groupby('model')['value'].sem().reset_index() * 2)
    g2 = sns.barplot(
        data=melted_df_keller,
        x="model", y="value",
        errorbar="se", palette="dark", alpha=.6, ax=ax)
    # g2.set_axis_labels("", title)
    g2.set(xlabel='Model', ylabel=title)
    # g2.despine(left=True)
    # g2.set_axis_labels("", "Body mass (g)")
    # g2.legend.set_title("")
    # g.set_xticklabels(tasks, rotation=45)
    # g2.figure.savefig(figure_name+".pdf")
def post_process_dataframe(corrss, msess, df_cor_pom, df_cor_alva, df_mse_pom, df_mse_alva, tasks,
                           figure_name="def"):
    plt.rcParams["font.size"] = 40
    corrss_1_12 = corrss.loc[((corrss["layer"] == 0) | (corrss["layer"] == 12)) & (corrss["model"] == "molformer")]
    del corrss_1_12["model"]
    melted_corrss_1_12 = corrss_1_12.melt(id_vars=['layer'], var_name='descritpor')
    melted_corrss_filtered_increasing = melted_corrss_1_12.groupby('descritpor').filter(
        lambda x: x.loc[x['layer'] == 12, 'value'].abs().mean() > x.loc[x['layer'] == 0, 'value'].abs().mean())
    melted_corrss_filtered_decreasing = melted_corrss_1_12.groupby('descritpor').filter(
        lambda x: x.loc[x['layer'] == 0, 'value'].abs().mean() > x.loc[x['layer'] == 12, 'value'].abs().mean())
    print(melted_corrss_1_12.descritpor.unique())
    melted_corrss_filtered_increasing['trend'] = 'Increasing'
    melted_corrss_filtered_decreasing['trend'] = 'Decreasing'
    melted_corrss_filtered = pd.concat((melted_corrss_filtered_increasing, melted_corrss_filtered_decreasing))
    # fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(30,10))
    # sns.lineplot(
    # data=melted_corrss_filtered_increasing, x="layer", y="value", hue="descritpor", err_style='bars',ax=ax[0]
    # )
    # sns.lineplot(
    # data=melted_corrss_filtered_decreasing, x="layer", y="value", hue="descritpor", err_style='bars',ax=ax[1]
    # )
    # f1, ax_agg = plt.subplots(1, 2,figsize=(20, 5))
    f2, ax = plt.subplots(2, 1, figsize=(22, 22))
    # combine_visualize_separate(corrss.loc[corrss["layer"]==12,].iloc[:,corrss.columns != 'layer'], df_cor_pom,df_cor_alva,tasks,ax_agg[0],'Correlation Coefficient',figure_name="Correlation_Avg_"+figure_name)
    g1 = combine_visualize(corrss.loc[corrss["layer"] == 12].iloc[:, corrss.columns != 'layer'], df_cor_pom,
                           df_cor_alva, tasks, ax[0], 'Correlation Coefficient',
                           figure_name="Correlation_" + figure_name)
    g1.set_xlabel('')
    # combine_visualize_separate(msess.loc[msess["layer"]==12].iloc[:,msess.columns != 'layer'], df_mse_pom,df_mse_alva,tasks,ax_agg[1],'MSE',type="mse",figure_name="MSE_Avg_"+figure_name)
    g2 = combine_visualize(msess.loc[msess["layer"] == 12].iloc[:, msess.columns != 'layer'], df_mse_pom,
                           df_mse_alva, tasks, ax[1], 'NRMSE', type="mse", figure_name="MSE__" + figure_name)
    g2.set_xlabel('Descriptor')
    g1.legend().set_title("Model")
    handles, labels = g1.get_legend_handles_labels()
    g1.get_legend().remove()
    g2.legend().set_title("Model")
    handles, labels = g2.get_legend_handles_labels()
    g2.get_legend().remove()
    print(labels)
    f2.subplots_adjust(bottom=0.2, left=0.1, right=0.95, top=0.95)
    labels = ['MoLFormer', 'Open-POM', 'DAM']
    f2.legend(handles, labels, ncol=3, columnspacing=1, prop={'size': 40}, handlelength=1.5, loc="lower center",
              borderpad=0.3,
              bbox_to_anchor=(0.52, -0.07),
              frameon=True, labelspacing=0.4, handletextpad=0.2, )
    # plt.legend(title='Smoker', loc='upper left',)
    plt.subplots_adjust(hspace=0.6)
    f2.savefig(figure_name + ".pdf", bbox_inches='tight')
    return melted_corrss_filtered
