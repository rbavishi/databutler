import unittest

from databutler.pat import astlib
from databutler.pat.utils import codeutils
import json


class TestCodeUtils(unittest.TestCase):
    def test_notebook_magics(self):
        notebook = '{"cells":[{"metadata":{"_uuid":"8f2839f25d086af736a60e9eeb907d3b93b6e0e5",' \
                   '"_cell_guid":"b1076dfc-b9ad-4769-8c92-a6c4dae69d19","trusted":true},"cell_type":"code",' \
                   '"source":"# This Python 3 environment comes with many helpful analytics libraries installed\\n# ' \
                   'It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\\n# For ' \
                   'example, here\'s several helpful packages to load\\n\\nimport numpy as np # linear ' \
                   'algebra\\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\\n\\n# Input ' \
                   'data files are available in the read-only \\"../input/\\" directory\\n# For example, running this ' \
                   '(by clicking run or pressing Shift+Enter) will list all files under the input ' \
                   'directory\\n\\nimport os\\nfor dirname, _, filenames in os.walk(\'/kaggle/input\'):\\n    for ' \
                   'filename in filenames:\\n        print(os.path.join(dirname, filename))\\n\\n# You can write up ' \
                   'to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create ' \
                   'a version using \\"Save & Run All\\" \\n# You can also write temporary files to /kaggle/temp/, ' \
                   'but they won\'t be saved outside of the current session","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"trusted":true},"cell_type":"code","source":"import matplotlib.pyplot as plt",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"import seaborn as sns","execution_count":null,"outputs":[]},{"metadata":{' \
                   '"trusted":true},"cell_type":"code","source":"%matplotlib inline","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"_uuid":"d629ff2d2480ee46fbb7e2d37f6b5fab8052498a",' \
                   '"_cell_guid":"79c7e3d0-c299-4dcb-8224-4455121ee9b0","trusted":true},"cell_type":"code",' \
                   '"source":"df_train = pd.read_csv(\'/kaggle/input/titanic/train.csv\')\\ndf_test = pd.read_csv(' \
                   '\'/kaggle/input/titanic/test.csv\')\\ndf_true = pd.read_csv(' \
                   '\'/kaggle/input/titanic/gender_submission.csv\')","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"trusted":true},"cell_type":"code","source":"df_test = pd.merge(df_test,df_true,' \
                   'on=\'PassengerId\',how=\'left\')\\n","execution_count":null,"outputs":[]},{"metadata":{' \
                   '"trusted":true},"cell_type":"code","source":"df_train.head()\\n","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code","source":"df_train.info()",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"sns.pairplot(df_train)","execution_count":null,"outputs":[]},{"metadata":{' \
                   '"trusted":true},"cell_type":"code","source":"df_test.head()","execution_count":null,"outputs":[' \
                   ']},{"metadata":{"trusted":true},"cell_type":"code","source":"df_pid = df_test[\'PassengerId\']",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"df_test.info()","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"df_true.head()","execution_count":null,"outputs":[]},{"metadata":{' \
                   '"trusted":true},"cell_type":"code","source":"df_test.drop(\'Name\',axis=1,inplace=True)",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"df_train.drop(\'Name\',axis=1,inplace=True)","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"trusted":true},"cell_type":"code","source":"len(df_train)","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code","source":"100*df_train.isna().sum(' \
                   ')/len(df_train)","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"100*df_test.isna().sum()/len(df_test)","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code","source":"df_test.isna().sum()",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"df_train.corr()[\'Age\'].sort_values()","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"trusted":true},"cell_type":"code","source":"df_test.drop(\'Cabin\',axis=1,' \
                   'inplace=True)\\ndf_train.drop(\'Cabin\',axis=1,inplace=True)","execution_count":null,"outputs":[' \
                   ']},{"metadata":{"trusted":true},"cell_type":"code","source":"df_train.drop(\'Ticket\',axis=1,' \
                   'inplace=True)","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"df_test.drop(\'Ticket\',axis=1,inplace=True)",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"df_test.drop(\'PassengerId\',axis=1,inplace=True)\\ndf_train.drop(\'PassengerId\',' \
                   'axis=1,inplace=True)","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"100*df_test.isna().sum()/len(df_test)","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code","source":"","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code","source":"100*df_train.isna().sum(' \
                   ')/len(df_train)","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"df_test.corr()[\'Age\'].sort_values()","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code","source":"sns.scatterplot(' \
                   'x=\'Age\',y=\'Fare\',data=df_test)","execution_count":null,"outputs":[]},{"metadata":{' \
                   '"trusted":true},"cell_type":"code","source":"df_train.describe()","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code","source":"df_train.head()",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"df_test.head()","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"df_test.describe()","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"trusted":true},"cell_type":"code","source":"df_train = pd.get_dummies(df_train, ' \
                   'columns=[\'Embarked\',\'Sex\'], drop_first=True)","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"trusted":true},"cell_type":"code","source":"df_test = pd.get_dummies(df_test, ' \
                   'columns=[\'Embarked\',\'Sex\'], drop_first=True)","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"trusted":true},"cell_type":"code","source":"df_test = df_test.reindex(' \
                   'df_train.columns, axis=1)\\n","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"df_train.isna().sum()","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"trusted":true},"cell_type":"code","source":"df_train.corr()[\'Age\'].sort_values(' \
                   ')","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"df_test.isna().sum()","execution_count":null,"outputs":[]},{"metadata":{' \
                   '"trusted":true},"cell_type":"code","source":"df_test[\'Fare\'].fillna((df_test[\'Fare\'].mean()), ' \
                   'inplace=True)","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"df_test.isna().sum()","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"trusted":true},"cell_type":"code","source":"from sklearn.linear_model import ' \
                   'LinearRegression","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"lnreg_age = LinearRegression()","execution_count":null,"outputs":[' \
                   ']},{"metadata":{"trusted":true},"cell_type":"code","source":"x_train, y_train = df_train.dropna(' \
                   ')[[\'Sex_male\',\'Fare\']], df_train.dropna()[[\'Age\']]","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"trusted":true},"cell_type":"code","source":"lnreg_age.fit(X=x_train,y=y_train)",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"df_train.loc[df_train[\'Age\'].isna(),\'Age\']=df_train.loc[df_train[\'Age\'].isna(' \
                   ')].apply((lambda row: lnreg_age.predict([[row[\'Sex_male\'],row[\'Fare\']]]).item(0)),axis=1)",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"df_test.loc[df_test[\'Age\'].isna(),\'Age\']=df_test.loc[df_test[\'Age\'].isna(' \
                   ')].apply((lambda row: lnreg_age.predict([[row[\'Sex_male\'],row[\'Fare\']]]).item(0)),axis=1)",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"df_train.isna().sum()","execution_count":null,"outputs":[]},{"metadata":{' \
                   '"trusted":true},"cell_type":"code","source":"df_test.isna().sum()","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code","source":"","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code","source":"df_train.head()",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"df_test.head()","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"from sklearn.ensemble import RandomForestClassifier",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"df_test[df_test.columns[1:]]","execution_count":null,"outputs":[]},{"metadata":{' \
                   '"trusted":true},"cell_type":"code","source":"X_train,y_train = df_train[df_train.columns[1:]],' \
                   'df_train[df_train.columns[0]]","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"X_test,y_test = df_test[df_test.columns[1:]],' \
                   'df_test[df_test.columns[0]]","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"from sklearn.preprocessing import MinMaxScaler",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"X_train","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"scaler = MinMaxScaler()","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"trusted":true},"cell_type":"code","source":"X_train = scaler.fit_transform(' \
                   'X_train)","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"X_train","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"X_test = scaler.transform(X_test)","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code","source":"rndf = ' \
                   'RandomForestClassifier(n_estimators=128)","execution_count":null,"outputs":[]},{"metadata":{' \
                   '"trusted":true},"cell_type":"code","source":"rndf.fit(X_train,y_train)","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code","source":"predict_y = rndf.predict(' \
                   'X_test)","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"from sklearn.metrics import confusion_matrix, classification_report",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"print(confusion_matrix(y_test,predict_y))\\nprint(classification_report(y_test,' \
                   'predict_y))","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"df_predict = pd.DataFrame(predict_y,columns=[\'Survived\'])",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"trusted":true},"cell_type":"code",' \
                   '"source":"df_pred_id = pd.concat([df_pid,df_predict],axis=1)","execution_count":null,"outputs":[' \
                   ']},{"metadata":{"trusted":true},"cell_type":"code","source":"df_pred_id.to_csv(' \
                   '\'submission_predict.csv\')","execution_count":null,"outputs":[]},{"metadata":{"trusted":true},' \
                   '"cell_type":"code","source":"","execution_count":null,"outputs":[]}],"metadata":{"kernelspec":{' \
                   '"language":"python","display_name":"Python 3","name":"python3"},"language_info":{' \
                   '"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4",' \
                   '"file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python",' \
                   '"mimetype":"text/x-python"}},"nbformat":4,"nbformat_minor":4} '
        json.loads(notebook)
        d = codeutils.convert_python_notebook_magics(notebook)
        s = codeutils.convert_python_notebook_to_code(d)
        self.assertIn("%matplotlib inline", repr(notebook))
        self.assertNotIn("%matplotlib inline", repr(d))
        self.assertNotIn("%matplotlib inline", s)
        self.assertIn("get_ipython", repr(d))
        self.assertIn("get_ipython", s)
        astlib.parse(d, extension='.ipynb')
        astlib.parse(s, extension='.py')

    def test_2(self):
        notebook = '{"cells":[{"metadata":{},"cell_type":"markdown","source":"# Titanic Models ' \
                   'Benchmark<br/><sup>Classification</sup>\\n\\n### **Dataset:**  [titanic](' \
                   'https://www.kaggle.com/c/titanic/data)\\n#### Regression Benchmark: [House Prices Models ' \
                   'Benchmark](https://www.kaggle.com/aravrs/house-prices-models-benchmark)\\n\\n<sup ' \
                   'style=\\"color:red;\\">Work in progess.</sup><br/>\\n\\n---"},{"metadata":{' \
                   '"_uuid":"d629ff2d2480ee46fbb7e2d37f6b5fab8052498a",' \
                   '"_cell_guid":"79c7e3d0-c299-4dcb-8224-4455121ee9b0","trusted":true,"_kg_hide-input":true,' \
                   '"_kg_hide-output":true},"cell_type":"code","source":"import os\\nimport time\\nimport ' \
                   'random\\n\\nimport numpy as np\\nimport pandas as pd\\nimport matplotlib.pyplot as plt\\nimport ' \
                   'seaborn as sns\\n\\nsns.set_palette(\'Set1\')\\nplt.rcParams[\'figure.figsize\'] = (20, ' \
                   '8)\\nplt.rcParams[\'figure.dpi\'] = 200\\n\\nimport warnings\\nwarnings.filterwarnings(' \
                   '\'ignore\')","execution_count":null,"outputs":[]},{"metadata":{},"cell_type":"markdown",' \
                   '"source":"### Load Data"},{"metadata":{"_kg_hide-input":true,"trusted":true},"cell_type":"code",' \
                   '"source":"DATA_DIR = \'../input/titanic/\'\\nos.listdir(DATA_DIR)\\n\\ntrain_df = pd.read_csv(' \
                   'DATA_DIR + \'train.csv\')\\ntest_df = pd.read_csv(DATA_DIR + \'test.csv\')\\nsub_df = ' \
                   'pd.read_csv(DATA_DIR + \'gender_submission.csv\')\\n\\nprint(\' Train:\', train_df.shape, ' \
                   '\' Test:\', test_df.shape, \' Sub:\', sub_df.shape)","execution_count":null,"outputs":[]},' \
                   '{"metadata":{},"cell_type":"markdown","source":"# Basic EDA"},{"metadata":{"_kg_hide-input":true,' \
                   '"trusted":true},"cell_type":"code","source":"fig = sns.heatmap(train_df.isnull(), cbar=False, ' \
                   'cmap=\'hot_r\', yticklabels=[]).set_title(\'Missing Values\', fontsize=24);",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"_kg_hide-input":true,"trusted":true},' \
                   '"cell_type":"code","source":"cols = [\'Survived\', \'Pclass\', \'Sex\', \'SibSp\', \'Parch\', ' \
                   '\'Embarked\']\\n\\nn_rows, n_cols = 2, 3\\nfig, axs = plt.subplots(n_rows, n_cols, ' \
                   'figsize=(n_cols*6, n_rows*5))\\nfig.suptitle(\'Count Plots\', fontsize=26, ' \
                   'y=1.05)\\n\\nsns.countplot(train_df[\'Survived\'], ax=axs[0][0]).set_title(\'Survived count\', ' \
                   'fontsize=20)\\nfor r in range(n_rows):\\n    for c in range(n_cols):\\n        if r!=0 or ' \
                   'c!=0:\\n            i = r*n_cols+c\\n            ax = axs[r][c]\\n            sns.countplot(' \
                   'train_df[cols[i]], hue=train_df[\'Survived\'], ax=ax)\\n            ax.set_title(cols[i]+\' ' \
                   'count\', fontsize=20)\\n            ax.legend(title=\'Survived\', loc=\'upper ' \
                   'right\')\\nplt.tight_layout()","execution_count":null,"outputs":[]},{"metadata":{' \
                   '"_kg_hide-input":true,"trusted":true},"cell_type":"code","source":"fig, ax = plt.subplots(1, ' \
                   '2)\\nfig.suptitle(\'Age Distribution\', fontsize=26)\\n\\nf = sns.distplot(train_df[\'Age\'], ' \
                   'color=\'g\', bins=40, ax=ax[0])\\n\\ng = sns.kdeplot(train_df[\'Age\'].loc[train_df[\'Survived\'] ' \
                   '== 0], \\n                shade=True, ax=ax[1], label=\'Not Survived\')\\ng = sns.kdeplot(' \
                   'train_df[\'Age\'].loc[train_df[\'Survived\'] == 1], \\n                shade=True, ax=ax[1], ' \
                   'label=\'Survived\').set_xlabel(\'Age\')","execution_count":null,"outputs":[]},{"metadata":{' \
                   '"_kg_hide-input":true,"trusted":true},"cell_type":"code","source":"fig, ax = plt.subplots(1, ' \
                   '2)\\nfig.suptitle(\'Fare Distribution\', fontsize=26)\\n\\nf = sns.distplot(train_df[\'Fare\'], ' \
                   'color=\'g\', bins=40, ax=ax[0])\\n\\ng = sns.kdeplot(train_df[\'Fare\'].loc[train_df[' \
                   '\'Survived\'] == 0], \\n                shade=True, ax=ax[1], label=\'Not Survived\')\\ng = ' \
                   'sns.kdeplot(train_df[\'Fare\'].loc[train_df[\'Survived\'] == 1], \\n                shade=True, ' \
                   'ax=ax[1], label=\'Survived\').set_xlabel(\'Fare\')","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"_kg_hide-input":true,"trusted":true},"cell_type":"code","source":"sns.heatmap(' \
                   'train_df.corr(), annot=True, cmap=\'RdBu_r\',\\n            center=0, vmin=-1, vmax=1, ' \
                   'linewidth=2, annot_kws={\\"fontsize\\":12},\\n            square=False, cbar=True).set_title(' \
                   '\'Correlation matrix\', fontsize=24);\\nplt.yticks(rotation=0);","execution_count":null,' \
                   '"outputs":[]},{"metadata":{},"cell_type":"markdown","source":"# Multi Model Benchmark"},' \
                   '{"metadata":{},"cell_type":"markdown","source":"Install the necessary libraries and setup the ' \
                   'environment"},{"metadata":{"_kg_hide-input":true,"_kg_hide-output":true,"trusted":true},' \
                   '"cell_type":"code","source":"!pip install pycaret -q\\nprint()","execution_count":null,' \
                   '"outputs":[]},' \
                   '{"metadata":{"_kg_hide-input":true,"trusted":true},"cell_type":"code","source":"from ' \
                   'pycaret.utils import version\\nfrom pycaret.classification import *\\nprint(\'Pycaret Verion:\', ' \
                   'version())","execution_count":null,"outputs":[]},{"metadata":{"_kg_hide-input":true,' \
                   '"_kg_hide-output":true,"trusted":true},"cell_type":"code","source":"clf = setup(train_df, ' \
                   'target=\'Survived\', session_id=42, log_experiment=True, experiment_name=\'titanic\', ' \
                   'silent=True)","execution_count":null,"outputs":[]},{"metadata":{},"cell_type":"markdown",' \
                   '"source":"Compare various models and find the best model"},{"metadata":{"_kg_hide-input":true,' \
                   '"trusted":true},"cell_type":"code","source":"models = compare_models(n_select=25)",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{},"cell_type":"markdown","source":"## Analyse ' \
                   'all models"},{"metadata":{"_kg_hide-input":true,"_kg_hide-output":true,"trusted":true},' \
                   '"cell_type":"code","source":"#### hacky\\n\\nplot_types = [\'auc\', \'threshold\', \'pr\', ' \
                   '\'confusion_matrix\', \'error\', \'class_report\', \'boundary\', \\n              \'learning\', ' \
                   '\'calibration\', \'vc\', \'feature\', \'gain\'] # \'lift\', \'rfe\'\\n\\n# to plot same plots for ' \
                   'different models\\ndef plot_util(models, plot, title=\'Comparison plot\'):\\n    imgs = []\\n    ' \
                   'for model in models:\\n        try: imgs.append(plt.imread(plot_model(model, plot=plot, ' \
                   'save=True)))\\n        except: imgs.append(np.ones((1100, 1600, 4)))\\n\\n    n_rows, ' \
                   'n_cols = len(imgs)//2, 2\\n    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, ' \
                   'n_rows*3))\\n    fig.suptitle(title, fontsize=12, y=0.89)\\n    \\n    for r in range(n_rows):\\n ' \
                   '       for c in range(n_cols):\\n            i = r*n_cols+c\\n            plt.subplot(n_rows, ' \
                   'n_cols, i+1)\\n            plt.imshow(imgs[i])\\n            plt.axis(\'off\')\\n    ' \
                   'fig.subplots_adjust(wspace=0, hspace=0)\\n    \\n# to plot all plots for same ' \
                   'model\\ndef model_all_plots(model, title=\'Model plot\'):\\n    imgs = []\\n    for plot in ' \
                   'plot_types:\\n        try: imgs.append(plt.imread(plot_model(model, plot=plot, save=True)))\\n    ' \
                   '    except: imgs.append(np.ones((1100, 1600, 4)))\\n    \\n    n_rows, n_cols = len(imgs)//2, ' \
                   '2\\n    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))\\n    fig.suptitle(' \
                   'title, fontsize=12, y=0.89)\\n\\n    for r in range(n_rows):\\n        for c in range(n_cols):\\n ' \
                   '           i = r*n_cols+c\\n            plt.subplot(n_rows, n_cols, i+1)\\n            ' \
                   'plt.imshow(imgs[i])\\n            plt.axis(\'off\')\\n    fig.subplots_adjust(wspace=0, ' \
                   'hspace=0)","execution_count":null,"outputs":[]},{"metadata":{},"cell_type":"markdown",' \
                   '"source":"Various analytical plots of models from best to worst. <br/>\\n> If a metric/plot is ' \
                   'not possible for a particular model, it\'s left blank."},{"metadata":{"_kg_hide-input":true,' \
                   '"_kg_hide-output":true,"trusted":true},"cell_type":"code","source":"# test plot\\n\\nplot_model(' \
                   'models[0], plot=\'dimension\')\\n\\n# plot params\\n\\nsns.set_palette(\'Set1\')\\nplt.rcParams[' \
                   '\'axes.titlesize\'] = 18\\nplt.rcParams[\'savefig.dpi\'] = 200\\nplt.rcParams[\'savefig.bbox\'] = ' \
                   '\'tight\'\\nplt.rcParams[\'savefig.pad_inches\'] = 0.3","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"_kg_hide-input":true,"trusted":true},"cell_type":"code","source":"plot_util(models, ' \
                   '\'auc\', \'ROC Curves Comparison Plot\')","execution_count":null,"outputs":[]},{"metadata":{' \
                   '"_kg_hide-input":true,"trusted":true},"cell_type":"code","source":"plot_util(models, ' \
                   '\'threshold\', \'Thresholds Comparison Plot\')","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"_kg_hide-input":true,"trusted":true},"cell_type":"code","source":"plot_util(models, ' \
                   '\'pr\', \'Precision-Recall Curve Comparison Plot\')","execution_count":null,"outputs":[]},' \
                   '{"metadata":{"_kg_hide-input":true,"trusted":true},"cell_type":"code","source":"plot_util(models, ' \
                   '\'confusion_matrix\', \'Confusion Matrix Comparison Plot\')","execution_count":null,"outputs":[' \
                   ']},{"metadata":{"_kg_hide-input":true,"trusted":true},"cell_type":"code","source":"plot_util(' \
                   'models, \'error\', \'Class Prediction Error Comparison Plot\')","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"_kg_hide-input":true,"trusted":true},"cell_type":"code",' \
                   '"source":"plot_util(models, \'class_report\', \'Classification Error Comparison\')",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"_kg_hide-input":true,"trusted":true},' \
                   '"cell_type":"code","source":"plot_util(models, \'boundary\', \'Boundaries Comparison Plot\')",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"_kg_hide-input":true,"trusted":true},' \
                   '"cell_type":"code","source":"plot_util(models, \'learning\', \'Learning Curve Comparison ' \
                   'Plot\')","execution_count":null,"outputs":[]},{"metadata":{"_kg_hide-input":true,"trusted":true},' \
                   '"cell_type":"code","source":"plot_util(models, \'calibration\', \'Calibration Comparasion ' \
                   'Plot\')","execution_count":null,"outputs":[]},{"metadata":{"_kg_hide-input":true,"trusted":true},' \
                   '"cell_type":"code","source":"plot_util(models, \'vc\', \'Validation Curve Comparison Plot\')",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"_kg_hide-input":true,"trusted":true},' \
                   '"cell_type":"code","source":"plot_util(models, \'feature\', \'Feature Importance Comparison ' \
                   'Plot\')","execution_count":null,"outputs":[]},{"metadata":{"_kg_hide-input":true,"trusted":true},' \
                   '"cell_type":"code","source":"plot_util(models, \'lift\', \'Lift Curve Comparison Plot\')",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"_kg_hide-input":true,"trusted":true},' \
                   '"cell_type":"code","source":"plot_util(models, \'gain\', \'Cumulative Gain Curve Comparison ' \
                   'Plot\')","execution_count":null,"outputs":[]},{"metadata":{"_kg_hide-input":true,"trusted":true},' \
                   '"cell_type":"code","source":"plot_model(models[0], plot=\'manifold\')","execution_count":null,' \
                   '"outputs":[]},{"metadata":{"_kg_hide-input":true,"trusted":true},"cell_type":"code",' \
                   '"source":"plot_model(models[0], plot=\'dimension\')","execution_count":null,"outputs":[]},' \
                   '{"metadata":{},"cell_type":"markdown","source":"## The Best Model"},{"metadata":{' \
                   '"_kg_hide-input":true,"_kg_hide-output":true,"trusted":true},"cell_type":"code",' \
                   '"source":"final_model = models[0]","execution_count":null,"outputs":[]},{"metadata":{' \
                   '"_kg_hide-input":true,"trusted":true},"cell_type":"code","source":"model_all_plots(final_model, ' \
                   '\'Final Model Plots\')","execution_count":null,"outputs":[]},{"metadata":{"_kg_hide-input":true,' \
                   '"trusted":true},"cell_type":"code","source":"interpret_model(final_model, plot=\'summary\')",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{},"cell_type":"markdown","source":"### Make ' \
                   'predictions"},{"metadata":{"_kg_hide-input":true,"_kg_hide-output":true,"trusted":true},' \
                   '"cell_type":"code","source":"predictions = predict_model(final_model, data=test_df)",' \
                   '"execution_count":null,"outputs":[]},{"metadata":{"_kg_hide-input":true,"_kg_hide-output":false,' \
                   '"trusted":true},"cell_type":"code","source":"submission = predictions[[\'PassengerId\', ' \
                   '\'Label\']].rename(columns={\'Label\': \'Survived\'})\\nsubmission.to_csv(\'submission.csv\', ' \
                   'index=False)\\nprint(\'Saved submission.csv\')\\nsubmission.head()","execution_count":null,' \
                   '"outputs":[]},{"metadata":{},"cell_type":"markdown","source":"---"}],"metadata":{"kernelspec":{' \
                   '"language":"python","display_name":"Python 3","name":"python3"},"language_info":{' \
                   '"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4",' \
                   '"file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python",' \
                   '"mimetype":"text/x-python"}},"nbformat":4,"nbformat_minor":4} '

        json.loads(notebook)
        d = codeutils.convert_python_notebook_magics(notebook)
        s = codeutils.convert_python_notebook_to_code(d)
        self.assertIn("!pip install", repr(notebook))
        self.assertNotIn("!pip install", repr(d))
        self.assertNotIn("!pip install", s)
        self.assertIn("get_ipython", repr(d))
        self.assertIn("get_ipython", s)
        self.assertIn("pip", repr(d))
        self.assertIn("pip", s)
        astlib.parse(d, extension='.ipynb')
        astlib.parse(s, extension='.py')


if __name__ == '__main__':
    unittest.main()
