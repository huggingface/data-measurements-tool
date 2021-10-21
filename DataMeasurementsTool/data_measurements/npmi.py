from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np


class nPMI():
    # TODO: Expand beyond pairwise
    def __init__(self, df, term_df):
        self.df = df
        self.term_df = term_df
        self.npmi_bias = None
        self.subgroup1 = None
        self.subgroup2 = None
        # Storing results
        self.paired_results = pd.DataFrame()

    def set_subgroups(self, subgroup1, subgroup2):
        self.subgroup1 = subgroup1
        self.subgroup2 = subgroup2

    def calc_cooccurrences(self, subgroup, mlb=None):
        if not mlb:
            mlb = MultiLabelBinarizer()
        # Makes a sparse vector (shape: # sentences x # words),
        # with the count of each word per sentence.
        df_mlb = pd.DataFrame(mlb.fit_transform(self.df['tokenized']))
        # Index of the subgroup word in the sparse vector
        subgroup_idx = np.where(mlb.classes_ == subgroup)[0][0]
        # Dataframe for the subgroup (with counts)
        df_subgroup = df_mlb.iloc[:, subgroup_idx]
        # Create cooccurence matrix for the given subgroup and all other words.
        # Note it also includes the word itself, so that count should maybe be subtracted
        # (the word will always co-occur with itself)
        df_coo = pd.DataFrame(df_mlb.T.dot(df_subgroup))#.drop(index=subgroup_idx, axis=1)
        return df_coo

    def calc_metrics(self, subgroup):
        mlb = MultiLabelBinarizer()
        df_coo = self.calc_cooccurrences(subgroup, mlb)
        count_df = self.get_count(df_coo, mlb)
        pmi_df = self.get_PMI(df_coo, subgroup, mlb)
        npmi_df = self.get_nPMI(pmi_df, df_coo, mlb)
        self.paired_results[subgroup + '-pmi']  = pmi_df['pmi']
        self.paired_results[subgroup + '-npmi'] = npmi_df['npmi']
        self.paired_results[subgroup + '-count'] = count_df['count']
        return self.paired_results

    def get_PMI(self, df_coo, subgroup, mlb):
        # PMI(x;y) = h(y) - h(y|x)
        #          = h(subgroup) - h(subgroup|word)
        #          = log (p(subgroup|word) / p(subgroup))
        # nPMI additionally divides by -log(p(x,y)) = -log(p(x|y)p(y))
        #
        # Calculation of p(subgroup)
        subgroup_prob = self.term_df.loc[subgroup]['proportion']
        # Apply a function to all words to calculate log p(subgroup|word)
        # The word is indexed by mlb.classes_ ;
        # we pull out the word using the mlb.classes_ index and then get its count using our main term_df
        # Calculation:
        # p(subgroup|word) = count(subgroup,word) / count(word)
        #                  = x.values             / term_df.loc[mlb.classes_[x.index]]['count']
        pmi_df = pd.DataFrame(df_coo.apply(lambda x: np.log(
            x.values/self.term_df.loc[mlb.classes_[x.index]][
                'count']/subgroup_prob)))
        pmi_df.columns = ['pmi']
        # If all went well, this will be correlated with high frequency words
        # Until normalizing
        # Note: A potentially faster solution for adding count, npmi, can be based on this:
        # #df_test['size_kb'],  df_test['size_mb'], df_test['size_gb'] = zip(*df_test['size'].apply(sizes))
        return pmi_df

    def get_nPMI(self, pmi_df, df_coo, mlb):
        normalize_df = pd.DataFrame(df_coo.apply(lambda x: -np.log(
            x.values/self.term_df.loc[mlb.classes_[x.index]]['count'] *
            self.term_df.loc[mlb.classes_[x.index]]['proportion'])))
        # npmi_df = pmi_df/normalize_df
        npmi_df = pd.DataFrame(pmi_df['pmi']/normalize_df[0])
        npmi_df.columns = ['npmi']
        return npmi_df

    def get_count(self, df_coo, mlb):
        count_df = pd.DataFrame(df_coo.apply(lambda x: pd.Series(x.values,
                                                                  mlb.classes_[x.index])))
        count_df.columns=['count']
        return count_df

    def calc_npmi_bias(self):
        if not self.npmi_bias:
            # woman - man: If it's negative, it's man-biased; if it's positive, it's woman positive.
            self.npmi_bias = self.paired_results[self.subgroup1 + '-npmi'] - \
                             self.paired_results[self.subgroup2 + '-npmi']
            self.paired_results['npmi_bias'] = self.npmi_bias.dropna()
            self.paired_results = self.paired_results.dropna()
            # TODO: Do we care about PMI skew?
            # pmi_bias = pd.DataFrame(pmi_df_pair[subgroup1] - pmi_df_pair[
            # subgroup2])
        return self.paired_results