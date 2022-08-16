import utils.dataset_utils as utils
import evaluate

LABEL_FIELD = "labels"
LABEL_NAMES = "label_names"
LABEL_LIST = "label_list"
LABEL_JSON = "labels.json"
LABEL_FIG_JSON = "labels_fig.json"
# Specific to the evaluate library
EVAL_LABEL_MEASURE = "label_distribution"
EVAL_LABEL_ID = "labels"
EVAL_LABEL_FRAC = "fractions"


class Labels:
    """
    Uses the Dataset to extract the label column and compute label measurements.
    """

    def __init__(self, dset, ds_name=None, config_name=None, label_field=None, label_names=None,
                 cache_path=None, use_cache=False, save=False):
        # Input HuggingFace Dataset.
        self.dset = dset
        # These are used to extract label names, when the label names
        # are stored in the Dataset object but not in the "label" column
        # we are working with, which may instead just be ints corresponding to
        # the names
        self.ds_name = ds_name
        self.config_name = config_name
        if not label_field:
            self.label_field = LABEL_FIELD
            print(
                "Name of the label field not provided; assuming %s " %
                LABEL_FIELD)
        self.label_names = label_names
        self.use_cache = use_cache
        self.cache_path = pjoin(cache_path, LABELS)
        # Filename for the figure
        self.labels_fig_json_fid = pjoin(self.cache_path, LABEL_FIG_JSON)
        # Filename for the measurement cache
        self.labels_json_fid = pjoin(self.cache_path, LABEL_JSON)
        # Values in the Dataset label column
        self.label_list = []
        # The names of the labels in the Dataset
        self.label_names = []
        # Label figure
        self.fig_labels = None
        # Whether to save results
        self.save = save

    def load_or_prepare_labels(self):
        """
        For the DMT, we only need the figure.
        This checks whether the figure exists, first.
        If it doesn't, it creates one.
        """
        # Bools to track whether the data is newly prepared,
        # in which case we may want to cache it.
        prepared_fig = False
        prepared_measurement = False
        if self.use_cache:
            # Figure exists.
            if exists(self.labels_fig_json_fid):
                self.fig_labels = utils.read_plotly(self.labels_fig_json_fid)
            # Measurements exist, just not the figure; make it
            elif exists(self.label_json_fid):
                label_json = utils.read_json(self.labels_json_fid)
                self.label_list = label_json[LABEL_LIST]
                self.label_names = label_json[LABEL_NAMES]
                self.fig_labels = make_label_fig(self.label_list,
                                                 self.label_names, results)
                # We have newly prepared this figure
                prepared_fig = True
        # If we have not gotten the figure, calculate afresh.
        # This happens either because the cache is not used,
        # Or because the figure is not there.
        if not self.fig_labels:
            label_measurement = self.prepare_labels()
            self.fig_labels = make_label_fig(self.label_list, self.label_names,
                                             label_measurement)
            prepared_measurement = True
            prepared_fig = True

        if self.save:
            # Create the cache path if it's not there.
            os.makedirs(self.cache_path, exist_ok=True)
            # If the measurement is newly calculated, save it.
            if prepared_measurement:
                utils.write_json(results, self.labels_json_fid)
            # If the figure is newly created, save it
            if prepared_fig:
                utils.write_plotly(results, self.labels_fig_json_fid)

    def prepare_labels(self):
        """ Uses the evaluate library to return the label distribution. """
        # The input Dataset object
        self.label_list = self.dset[self.label_field]
        # Have to extract the label names from the Dataset object when the
        # actual dataset columns are just ints representing the label names.
        ds_name_to_dict = dataset_utils.get_dataset_info_dicts(self.ds_name)
        self.label_names = self.get_label_names(LABEL_FIELD, ds_name_to_dict, self.ds_name, self.config_name)
        label_distribution = evaluate.load(EVAL_LABEL_MEASURE)
        label_measurement = label_distribution.compute(data=self.label_list)
        return label_measurement

    def load_labels(self):
        results = utils.read_json(self.labels_json_fid)
        return results

    def get_label_names(self, label_field, ds_name_to_dict, ds_name, config_name):
        if label_field:
            label_field, label_names = (
                ds_name_to_dict[ds_name][config_name]["features"][label_field][
                    0]
                if len(ds_name_to_dict[ds_name][config_name]["features"][
                           label_field]) > 0
                else ((), [])
            )
        return label_names


def make_label_fig(label_list, label_names, results, chart_type="pie"):
    if chart_type == "bar":
        fig_labels = plt.bar(results[EVAL_LABEL_MEASURE][EVAL_LABEL_ID],
                             results[EVAL_LABEL_MEASURE][EVAL_LABEL_FRAC])
    else:
        if chart_type != "pie":
            print("Oops! Don't have that chart-type implemented.")
            print("Making the default pie chart")
        fig_labels = px.pie(label_list,
                            values=results[EVAL_LABEL_MEASURE][EVAL_LABEL_ID],
                            names=label_names)
        #     labels = label_df[label_field].unique()
        #     label_sums = [len(label_df[label_df[label_field] == label]) for label in labels]
        #     fig_labels = px.pie(label_df, values=label_sums, names=label_names)
        #     return fig_labels
    return fig_labels
