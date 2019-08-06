
# import numpy as np
#
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse
#
#
# def plot_point_cov(points, nstd=2, ax=None, **kwargs):
#     """
#     Plots an `nstd` sigma ellipse based on the mean and covariance of a point
#     "cloud" (points, an Nx2 array).
#     Parameters
#     ----------
#         points : An Nx2 array of the data points.
#         nstd : The radius of the ellipse in numbers of standard deviations.
#             Defaults to 2 standard deviations.
#         ax : The axis that the ellipse will be plotted on. Defaults to the
#             current axis.
#         Additional keyword arguments are pass on to the ellipse patch.
#     Returns
#     -------
#         A matplotlib ellipse artist
#     """
#     pos = points.mean(axis=0)
#     cov = np.cov(points, rowvar=False)
#     return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)
#
#
# def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
#     """
#     Plots an `nstd` sigma error ellipse based on the specified covariance
#     matrix (`cov`). Additional keyword arguments are passed on to the
#     ellipse patch artist.
#     Parameters
#     ----------
#         cov : The 2x2 covariance matrix to base the ellipse on
#         pos : The location of the center of the ellipse. Expects a 2-element
#             sequence of [x0, y0].
#         nstd : The radius of the ellipse in numbers of standard deviations.
#             Defaults to 2 standard deviations.
#         ax : The axis that the ellipse will be plotted on. Defaults to the
#             current axis.
#         Additional keyword arguments are pass on to the ellipse patch.
#     Returns
#     -------
#         A matplotlib ellipse artist
#     """
#
#     def eigsorted(cov):
#         vals, vecs = np.linalg.eigh(cov)
#         order = vals.argsort()[::-1]
#         return vals[order], vecs[:, order]
#
#     if ax is None:
#         ax = plt.gca()
#
#     vals, vecs = eigsorted(cov)
#     theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
#
#     # Width and height are "full" widths, not radius
#     width, height = 2 * nstd * np.sqrt(vals)
#     ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
#
#     ax.add_artist(ellip)
#     return ellip

#
# # if __name__ == '__main__':
# #     # -- Example usage -----------------------
# #     # Generate some random, correlated data
# #     points = np.random.multivariate_normal(
# #         mean=(0, 0), cov=[[5, .1], [0.1, 5]], size=1000
# #     )
# #     # Plot the raw points...
# #     x, y = points.T
# #     plt.plot(x, y, 'ro')
# #
# #     # Plot a transparent 3 standard deviation covariance ellipse
# #     plot_point_cov(points, nstd=3, alpha=0.5, color='green')
# #
# #     plt.show()


# Code to create visualizations of 2D embeddings


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns




# data is 2D array, two columns are the two dimensions
# labels is 1D array with the image labels
def plot_visualizations(data,labels):

    # Set style of scatterplot
    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")

    # Reshape labels array into (# of labels, 1) array
    labels = np.reshape(labels,(labels.__len__(),1))
    # Add labels to data array for a (# of labels, 3) array
    data = np.concatenate((data,labels),axis=1)

    #sns.set(rc={'figure.figsize': (15, 8.27)})

    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns = ['Dim1', 'Dim2','label'])

    # Create scatterplot of dataframe
    sns.lmplot(x="Dim1", y="Dim2",hue='label',ci=95,legend=False,
                   fit_reg= False,scatter_kws={"s":200, "alpha":0.2},data=df)
    # replace labels
    new_labels = ['0', '1','2','3','4','5','6','7','8','9']
    # plt.legend(bbox_to_anchor=(1.05, 0.5), title='', labels=new_labels)
    # plt.legend(loc='best')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1,labels=new_labels)

    #plt.title(r'$\beta = 10^{-3}$', weight='bold').set_fontsize('14')
    plt.xlabel('Dimension 1', weight='bold').set_fontsize('10')
    plt.ylabel('Dimension 2', weight='bold').set_fontsize('10')
    plt.show()


data = np.load('data.npy')
labels = np.load('labels.npy')
plot_visualizations(data,labels)