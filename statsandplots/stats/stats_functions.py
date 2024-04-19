# def pca_analysis(df, list_of_features, list_of_groups, components):
#     # extract features for PCA. All data should have mean=0 and variance=1
#     df1 = df.dropna(subset=list_of_features)
#     x = df1.loc[:, list_of_features].values
#     y = df1.loc[:, list_of_groups].values

#     # Stardarize the features
#     # x = StandardScaler().fit_transform(x)

#     # Create a covariance matrix
#     cov_data = np.corrcoef(x)

#     # Run the PCA analysis
#     pca = PCA(n_components=components)
#     principalComponents = pca.fit_transform(x)
#     principalDf = pd.DataFrame(
#         data=principalComponents,
#         columns=["principal component 1", "principal component 2"],
#     )
#     finalDf = pd.concat([principalDf, df1[groups]], axis=1)
#     explained_variance = pca.explained_variance_ratio_

#     if len(list_of_groups) > 1:
#         hue = list_of_groups[0]
#         split = list_of_groups[1]
#     else:
#         hue = list_of_groups[0]
#         split = None

#     # plot the PCA
#     sns.relplot(
#         data=finalDf,
#         x="principal component 1",
#         y="principal component 2",
#         hue=hue,
#         style=split,
#     )
#     return explained_variance
