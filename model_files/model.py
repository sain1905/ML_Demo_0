import joblib
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import time

#Function to Featurize Data
def featurize(X, encoder_list, best_features, two_feat_selc, three_feat_selc, pca, tSVD, F_ica, Dim_redn_corr_feat):
    ''' This function transforms categorical data to
    target encoded features, adds best feature and two & three way interaction features '''

    X0_encdr = encoder_list[0]
    X1_encdr = encoder_list[1]
    X2_encdr = encoder_list[2]
    X3_encdr = encoder_list[3]
    X5_encdr = encoder_list[4]
    X6_encdr = encoder_list[5]
    X8_encdr = encoder_list[6]

    X0_tgenc = X0_encdr.transform(X['X0'])
    X1_tgenc = X1_encdr.transform(X['X1'])
    X2_tgenc = X2_encdr.transform(X['X2'])
    X3_tgenc = X3_encdr.transform(X['X3'])
    X5_tgenc = X5_encdr.transform(X['X5'])
    X6_tgenc = X6_encdr.transform(X['X6'])
    X8_tgenc = X8_encdr.transform(X['X8'])

    X_1 = pd.concat([X0_tgenc,
                     X1_tgenc,
                     X2_tgenc,
                     X3_tgenc,
                     X5_tgenc,
                     X6_tgenc,
                     X8_tgenc,
                     X[best_features]], axis=1)

    # Generating dataframe for selected 2 feature combinations
    two_feat_dict = dict()
    for feat in two_feat_selc:
        ft_list = feat.split('_')
        two_feat_dict[feat] = X[ft_list[0]] + X[ft_list[1]]
    two_feat_df = pd.DataFrame(two_feat_dict)

    # Generating dataframe for selected 3 feature combinations
    three_feat_dict = dict()
    for feat in three_feat_selc:
        ft_list = feat.split('_')
        three_feat_dict[feat] = X[ft_list[0]] + X[ft_list[1]] + X[ft_list[2]]
    three_feat_df = pd.DataFrame(three_feat_dict)

    X_2 = pd.concat([X_1,
                     two_feat_df,
                     three_feat_df], axis=1)

    # Transforming target encoded features and best feature to Dimensional Reduction features
    X_pca = pca.transform(X_1)
    X_pca_df = pd.DataFrame(X_pca, index=list(X_1.index))
    X_pca_df.columns = [('pca_' + str(i)) for i in range(20)]

    X_tSVD = tSVD.transform(X_1)
    X_tSVD_df = pd.DataFrame(X_tSVD, index=list(X_1.index))
    X_tSVD_df.columns = [('tSVD_' + str(i)) for i in range(20)]

    X_Fica = F_ica.transform(X_1)
    X_Fica_df = pd.DataFrame(X_Fica, index=list(X_1.index))
    X_Fica_df.columns = [('ica_' + str(i)) for i in range(20)]

    X_3 = pd.concat([X_pca_df,
                     X_tSVD_df,
                     X_Fica_df, ], axis=1)

    X_3 = X_3.drop(Dim_redn_corr_feat, axis=1)

    X_mod = pd.concat([X_2,
                       X_3, ], axis=1)
    return X_mod

#To predict Testing time for given set of car configuration
def predict(X):
    '''
    This function predicts testing time of car for given configuration.
    Prediction is done after following transformation of data :
       - Removes the non important columns
       - Target encoders convert the categorical features to numerical values
       - pca, tSVD and F_ica does dimensionality reduction on the modified data.
       - adds two and three feature interactions
    '''
    start = time.time()
    # Verify columns size
    # Data columns with prediction values y
    cols1 = ['ID', 'y', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16',
             'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X26', 'X27', 'X28', 'X29', 'X30', 'X31', 'X32',
             'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39', 'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47',
             'X48', 'X49', 'X50', 'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'X60', 'X61', 'X62',
             'X63', 'X64', 'X65', 'X66', 'X67', 'X68', 'X69', 'X70', 'X71', 'X73', 'X74', 'X75', 'X76', 'X77', 'X78',
             'X79', 'X80', 'X81', 'X82', 'X83', 'X84', 'X85', 'X86', 'X87', 'X88', 'X89', 'X90', 'X91', 'X92', 'X93',
             'X94', 'X95', 'X96', 'X97', 'X98', 'X99', 'X100', 'X101', 'X102', 'X103', 'X104', 'X105', 'X106', 'X107',
             'X108', 'X109', 'X110', 'X111', 'X112', 'X113', 'X114', 'X115', 'X116', 'X117', 'X118', 'X119', 'X120',
             'X122', 'X123', 'X124', 'X125', 'X126', 'X127', 'X128', 'X129', 'X130', 'X131', 'X132', 'X133', 'X134',
             'X135', 'X136', 'X137', 'X138', 'X139', 'X140', 'X141', 'X142', 'X143', 'X144', 'X145', 'X146', 'X147',
             'X148', 'X150', 'X151', 'X152', 'X153', 'X154', 'X155', 'X156', 'X157', 'X158', 'X159', 'X160', 'X161',
             'X162', 'X163', 'X164', 'X165', 'X166', 'X167', 'X168', 'X169', 'X170', 'X171', 'X172', 'X173', 'X174',
             'X175', 'X176', 'X177', 'X178', 'X179', 'X180', 'X181', 'X182', 'X183', 'X184', 'X185', 'X186', 'X187',
             'X189', 'X190', 'X191', 'X192', 'X194', 'X195', 'X196', 'X197', 'X198', 'X199', 'X200', 'X201', 'X202',
             'X203', 'X204', 'X205', 'X206', 'X207', 'X208', 'X209', 'X210', 'X211', 'X212', 'X213', 'X214', 'X215',
             'X216', 'X217', 'X218', 'X219', 'X220', 'X221', 'X222', 'X223', 'X224', 'X225', 'X226', 'X227', 'X228',
             'X229', 'X230', 'X231', 'X232', 'X233', 'X234', 'X235', 'X236', 'X237', 'X238', 'X239', 'X240', 'X241',
             'X242', 'X243', 'X244', 'X245', 'X246', 'X247', 'X248', 'X249', 'X250', 'X251', 'X252', 'X253', 'X254',
             'X255', 'X256', 'X257', 'X258', 'X259', 'X260', 'X261', 'X262', 'X263', 'X264', 'X265', 'X266', 'X267',
             'X268', 'X269', 'X270', 'X271', 'X272', 'X273', 'X274', 'X275', 'X276', 'X277', 'X278', 'X279', 'X280',
             'X281', 'X282', 'X283', 'X284', 'X285', 'X286', 'X287', 'X288', 'X289', 'X290', 'X291', 'X292', 'X293',
             'X294', 'X295', 'X296', 'X297', 'X298', 'X299', 'X300', 'X301', 'X302', 'X304', 'X305', 'X306', 'X307',
             'X308', 'X309', 'X310', 'X311', 'X312', 'X313', 'X314', 'X315', 'X316', 'X317', 'X318', 'X319', 'X320',
             'X321', 'X322', 'X323', 'X324', 'X325', 'X326', 'X327', 'X328', 'X329', 'X330', 'X331', 'X332', 'X333',
             'X334', 'X335', 'X336', 'X337', 'X338', 'X339', 'X340', 'X341', 'X342', 'X343', 'X344', 'X345', 'X346',
             'X347', 'X348', 'X349', 'X350', 'X351', 'X352', 'X353', 'X354', 'X355', 'X356', 'X357', 'X358', 'X359',
             'X360', 'X361', 'X362', 'X363', 'X364', 'X365', 'X366', 'X367', 'X368', 'X369', 'X370', 'X371', 'X372',
             'X373', 'X374', 'X375', 'X376', 'X377', 'X378', 'X379', 'X380', 'X382', 'X383', 'X384', 'X385']

    # Data columns without prediction values y
    cols2 = ['ID', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16',
             'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X26', 'X27', 'X28', 'X29', 'X30', 'X31', 'X32',
             'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39', 'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47',
             'X48', 'X49', 'X50', 'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'X60', 'X61', 'X62',
             'X63', 'X64', 'X65', 'X66', 'X67', 'X68', 'X69', 'X70', 'X71', 'X73', 'X74', 'X75', 'X76', 'X77', 'X78',
             'X79', 'X80', 'X81', 'X82', 'X83', 'X84', 'X85', 'X86', 'X87', 'X88', 'X89', 'X90', 'X91', 'X92', 'X93',
             'X94', 'X95', 'X96', 'X97', 'X98', 'X99', 'X100', 'X101', 'X102', 'X103', 'X104', 'X105', 'X106', 'X107',
             'X108', 'X109', 'X110', 'X111', 'X112', 'X113', 'X114', 'X115', 'X116', 'X117', 'X118', 'X119', 'X120',
             'X122', 'X123', 'X124', 'X125', 'X126', 'X127', 'X128', 'X129', 'X130', 'X131', 'X132', 'X133', 'X134',
             'X135', 'X136', 'X137', 'X138', 'X139', 'X140', 'X141', 'X142', 'X143', 'X144', 'X145', 'X146', 'X147',
             'X148', 'X150', 'X151', 'X152', 'X153', 'X154', 'X155', 'X156', 'X157', 'X158', 'X159', 'X160', 'X161',
             'X162', 'X163', 'X164', 'X165', 'X166', 'X167', 'X168', 'X169', 'X170', 'X171', 'X172', 'X173', 'X174',
             'X175', 'X176', 'X177', 'X178', 'X179', 'X180', 'X181', 'X182', 'X183', 'X184', 'X185', 'X186', 'X187',
             'X189', 'X190', 'X191', 'X192', 'X194', 'X195', 'X196', 'X197', 'X198', 'X199', 'X200', 'X201', 'X202',
             'X203', 'X204', 'X205', 'X206', 'X207', 'X208', 'X209', 'X210', 'X211', 'X212', 'X213', 'X214', 'X215',
             'X216', 'X217', 'X218', 'X219', 'X220', 'X221', 'X222', 'X223', 'X224', 'X225', 'X226', 'X227', 'X228',
             'X229', 'X230', 'X231', 'X232', 'X233', 'X234', 'X235', 'X236', 'X237', 'X238', 'X239', 'X240', 'X241',
             'X242', 'X243', 'X244', 'X245', 'X246', 'X247', 'X248', 'X249', 'X250', 'X251', 'X252', 'X253', 'X254',
             'X255', 'X256', 'X257', 'X258', 'X259', 'X260', 'X261', 'X262', 'X263', 'X264', 'X265', 'X266', 'X267',
             'X268', 'X269', 'X270', 'X271', 'X272', 'X273', 'X274', 'X275', 'X276', 'X277', 'X278', 'X279', 'X280',
             'X281', 'X282', 'X283', 'X284', 'X285', 'X286', 'X287', 'X288', 'X289', 'X290', 'X291', 'X292', 'X293',
             'X294', 'X295', 'X296', 'X297', 'X298', 'X299', 'X300', 'X301', 'X302', 'X304', 'X305', 'X306', 'X307',
             'X308', 'X309', 'X310', 'X311', 'X312', 'X313', 'X314', 'X315', 'X316', 'X317', 'X318', 'X319', 'X320',
             'X321', 'X322', 'X323', 'X324', 'X325', 'X326', 'X327', 'X328', 'X329', 'X330', 'X331', 'X332', 'X333',
             'X334', 'X335', 'X336', 'X337', 'X338', 'X339', 'X340', 'X341', 'X342', 'X343', 'X344', 'X345', 'X346',
             'X347', 'X348', 'X349', 'X350', 'X351', 'X352', 'X353', 'X354', 'X355', 'X356', 'X357', 'X358', 'X359',
             'X360', 'X361', 'X362', 'X363', 'X364', 'X365', 'X366', 'X367', 'X368', 'X369', 'X370', 'X371', 'X372',
             'X373', 'X374', 'X375', 'X376', 'X377', 'X378', 'X379', 'X380', 'X382', 'X383', 'X384', 'X385']

    if ((X.columns.shape[0] == 378 and all(X.columns == cols1))
            or (X.columns.shape[0] == 377 and all(X.columns == cols2))):

        # Feature Filtering
        # Remove ID,X4 and Duplicate columns
        # Drop low variance features(<0.05)
        # Drop Correlated features
        # Selecting Best features takes care of above points
        best_features = ['X189', 'X261', 'X127', 'X313', 'X316', 'X29', 'X136', 'X128', 'X191', 'X350', 'X157', 'X159',
                         'X234', 'X275', 'X171', 'X156', 'X178', 'X31', 'X315', 'X118', 'X48', 'X148', 'X223', 'X103',
                         'X185', 'X28', 'X43', 'X334', 'X186', 'X224', 'X187', 'X14', 'X132', 'X51', 'X80', 'X256',
                         'X46',
                         'X108', 'X300', 'X286', 'X47', 'X218', 'X180', 'X337', 'X100', 'X64', 'X377', 'X354', 'X168',
                         'X158',
                         'X163', 'X273', 'X114', 'X246', 'X304', 'X351', 'X179', 'X45', 'X115', 'X155', 'X19', 'X202',
                         'X12',
                         'X126', 'X49', 'X355', 'X349', 'X336', 'X151', 'X98', 'X331', 'X81', 'X142', 'X130', 'X306',
                         'X22',
                         'X69', 'X96', 'X111', 'X220', 'X376', 'X68', 'X13', 'X75', 'X85', 'X343', 'X255', 'X203',
                         'X71',
                         'X197', 'X61', 'X301', 'X285', 'X27', 'X208']

        # Feature modifications
        # Target encoding
        encoder_list = joblib.load('./model_files/joblib_files/target_encoders.joblib')

        # Add Interaction feature columns
        two_feat_selc = ['X136_X314',
                         'X263_X314',
                         'X314_X315']

        three_feat_selc = ['X47_X263_X314',
                           'X62_X263_X314',
                           'X136_X221_X314',
                           'X136_X261_X315',
                           'X136_X314_X315',
                           'X221_X263_X314',
                           'X261_X263_X315',
                           'X263_X267_X314',
                           'X263_X314_X315',
                           'X263_X314_X344']

        # Dimensionality reduction methods fit on train data
        pca = joblib.load('./model_files/joblib_files/pca_transformer.joblib')
        tSVD = joblib.load('./model_files/joblib_files/tSVD_transformer.joblib')
        F_ica = joblib.load('./model_files/joblib_files/F_ica_transformer.joblib')

        # tSVD components which have correlation with other dimensionality reduction features
        # To be removed after dimention reduction
        Dim_redn_corr_feat = ['tSVD_0',
                              'tSVD_1',
                              'tSVD_2',
                              'tSVD_3',
                              'tSVD_7',
                              'tSVD_9',
                              'tSVD_10',
                              'tSVD_11',
                              'tSVD_12',
                              'tSVD_14',
                              'tSVD_15',
                              'tSVD_17',
                              'tSVD_18',
                              'tSVD_19']

        # To Get Dimentionality reduction features
        X_mod = featurize(X,
                          encoder_list,
                          best_features,
                          two_feat_selc,
                          three_feat_selc,
                          pca,
                          tSVD,
                          F_ica,
                          Dim_redn_corr_feat)

        # load saved model
        Model = joblib.load('./model_files/joblib_files/Best_model_EXT_F3F4_i0.21.joblib')
        #         print('Predictor Model :',Model)

        # Predict y
        y = Model.predict(X_mod)

        stop = time.time()
        print('Processing time (sec): ', stop - start)

        return y

    else:
        print('Data features size or the columns do not meet the requirement')
        return -1