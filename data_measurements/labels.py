from .dataset_utils import read_plotly


class Labels:
    def __init__(self, dset, label_field="label", label_names=None, cache_path=None, use_cache=False):
        self.dset = dset
        # "label" is assumed as default; most common column name for labels.
        self.label_field = label_field
        self.label_names = label_names
        self.use_cache = use_cache
        self.cache_path = pjoin(cache_path, "labels")
        os.makedirs(self.cache_path, exist_ok=True)
        self.labels_fig_json_fid = pjoin(self.cache_path, "labels_fig.json")
        self.labels_json_fid = pjoin(self.cache_path, "labels.json")
        self.label_list = []
        self.label_names = []
        # Label figure
        self.fig_labels = None

    def load_or_prepare_labels(self):
        """
        For the DMT, we only need the figure.
        This checks whether the figure exists, first.
        If it doesn't, it creates one.
        """
        if self.use_cache:
            self.load_files()
        else:
            self.label_list = self.prepare_labels()
            self.fig_labels = make_fig_labels(self.label_list, self.label_names, results)

    def load_files(self):
        if exists(self.labels_fig_json_fid):
            self.fig_labels = read_plotly(self.labels_fig_json_fid)
        elif exists(self.label_json_fid):
            label_json = read_json(self.labels_json_fid)
            self.label_list = label_json["labels"]
            self.label_names = label_json["label names"]
            self.fig_labels = make_fig_labels(self.label_list, self.label_names,
                                              results)

    def prepare_labels(self):
        """ Uses the evaluate library to return the label distribution. """
        self.label_list = self.dset[self.label_field]
        self.label_names = set(label_list)
        label_distribution = evaluate.load("label_distribution")
        results = label_distribution.compute(data=self.label_list)
        return results

def make_label_fig(label_list, label_names, results):
    #self.fig_labels = plt.bar(results['label_distribution']['labels'],
    #                          results['label_distribution']['fractions'])
    label_fig = px.pie(label_list,
                             values=results['label_distribution']['labels'],
                             names=label_names)
    return label_fig