import argparse
import json
import textwrap
from os import mkdir, getenv
from os.path import join as pjoin, isdir
from pathlib import Path
from dotenv import load_dotenv

from data_measurements import dataset_statistics
from data_measurements import dataset_utils
from huggingface_hub import create_repo, Repository
import smtplib

if Path(".env").is_file():
    load_dotenv(".env")

HF_TOKEN = getenv("HF_TOKEN")
EMAIL_PASSWORD = getenv("EMAIL_PASSWORD")

def load_or_prepare_widgets(ds_args, show_embeddings=False, use_cache=False):
    """
    Loader specifically for the widgets used in the app.
    Args:
        ds_args:
        show_embeddings:
        use_cache:

    Returns:

    """

    if not isdir(ds_args["cache_dir"]):
        print("Creating cache")
        # We need to preprocess everything.
        # This should eventually all go into a prepare_dataset CLI
        mkdir(ds_args["cache_dir"])


    dstats = dataset_statistics.DatasetStatisticsCacheClass(**ds_args,
                                                            use_cache=use_cache)
    # Header widget
    dstats.load_or_prepare_dset_peek()
    # General stats widget
    dstats.load_or_prepare_general_stats()
    # Labels widget
    try:
        dstats.set_label_field("label")
        dstats.load_or_prepare_labels()
    except:
        pass
    # Text lengths widget
    dstats.load_or_prepare_text_lengths()
    if show_embeddings:
        # Embeddings widget
        dstats.load_or_prepare_embeddings()
    # Text duplicates widget
    dstats.load_or_prepare_text_duplicates()
    # nPMI widget
    dstats.load_or_prepare_npmi()
    npmi_stats = dstats.npmi_stats
    # Handling for all pairs; in the UI, people select.
    do_npmi(npmi_stats)
    # Zipf widget
    dstats.load_or_prepare_zipf()


def load_or_prepare(dataset_args, do_html=False, use_cache=False):
    all = False
    dstats = dataset_statistics.DatasetStatisticsCacheClass(**dataset_args, use_cache=use_cache)
    print("Loading dataset.")
    dstats.load_or_prepare_dataset()
    print("Dataset loaded.  Preparing vocab.")
    dstats.load_or_prepare_vocab()
    print("Vocab prepared.")

    if not dataset_args["calculation"]:
        all = True

    if all or dataset_args["calculation"] == "general":
        print("\n* Calculating general statistics.")
        dstats.load_or_prepare_general_stats()
        print("Done!")
        print("Basic text statistics now available at %s." % dstats.general_stats_json_fid)
        print(
            "Text duplicates now available at %s." % dstats.dup_counts_df_fid
        )

    if all or dataset_args["calculation"] == "lengths":
        print("\n* Calculating text lengths.")
        fig_tok_length_fid = pjoin(dstats.cache_path, "lengths_fig.html")
        tok_length_json_fid = pjoin(dstats.cache_path, "lengths.json")
        dstats.load_or_prepare_text_lengths()
        with open(tok_length_json_fid, "w+") as f:
            json.dump(dstats.fig_tok_length.to_json(), f)
            print("Token lengths now available at %s." % tok_length_json_fid)
        if do_html:
            dstats.fig_tok_length.write_html(fig_tok_length_fid)
            print("Figure saved to %s." % fig_tok_length_fid)
        print("Done!")

    if all or dataset_args["calculation"] == "labels":
        if not dstats.label_field:
            print("Warning: You asked for label calculation, but didn't provide "
                  "the labels field name.  Assuming it is 'label'...")
            dstats.set_label_field("label")
            print("\n* Calculating label distribution.")
            dstats.load_or_prepare_labels()
            fig_label_html = pjoin(dstats.cache_path, "labels_fig.html")
            fig_label_json = pjoin(dstats.cache_path, "labels.json")
            dstats.fig_labels.write_html(fig_label_html)
            with open(fig_label_json, "w+") as f:
                json.dump(dstats.fig_labels.to_json(), f)
            print("Done!")
            print("Label distribution now available at %s." % dstats.label_dset_fid)
            print("Figure saved to %s." % fig_label_html)

    if all or dataset_args["calculation"] == "npmi":
        print("\n* Preparing nPMI.")
        npmi_stats = dataset_statistics.nPMIStatisticsCacheClass(
            dstats, use_cache=use_cache
        )
        do_npmi(npmi_stats, use_cache=use_cache)
        print("Done!")
        print(
            "nPMI results now available in %s for all identity terms that "
            "occur more than 10 times and all words that "
            "co-occur with both terms."
            % npmi_stats.pmi_cache_path
        )

    if all or dataset_args["calculation"] == "zipf":
        print("\n* Preparing Zipf.")
        zipf_fig_fid = pjoin(dstats.cache_path, "zipf_fig.html")
        zipf_json_fid = pjoin(dstats.cache_path, "zipf_fig.json")
        dstats.load_or_prepare_zipf()
        zipf_fig = dstats.zipf_fig
        with open(zipf_json_fid, "w+") as f:
            json.dump(zipf_fig.to_json(), f)
        zipf_fig.write_html(zipf_fig_fid)
        print("Done!")
        print("Zipf results now available at %s." % dstats.zipf_fid)
        print(
            "Figure saved to %s, with corresponding json at %s."
            % (zipf_fig_fid, zipf_json_fid)
        )

    # Don't do this one until someone specifically asks for it -- takes awhile.
    if dataset_args["calculation"] == "embeddings":
        print("\n* Preparing text embeddings.")
        dstats.load_or_prepare_embeddings()


def do_npmi(npmi_stats, use_cache=True):
    available_terms = npmi_stats.load_or_prepare_npmi_terms()
    completed_pairs = {}
    print("Iterating through terms for joint npmi.")
    for term1 in available_terms:
        for term2 in available_terms:
            if term1 != term2:
                sorted_terms = tuple(sorted([term1, term2]))
                if sorted_terms not in completed_pairs:
                    term1, term2 = sorted_terms
                    print("Computing nPMI statistics for %s and %s" % (term1, term2))
                    _ = npmi_stats.load_or_prepare_joint_npmi(sorted_terms)
                    completed_pairs[tuple(sorted_terms)] = {}


def get_text_label_df(
    ds_name,
    config_name,
    split_name,
    text_field,
    label_field,
    calculation,
    out_dir,
    do_html=False,
    use_cache=True,
):
    if not use_cache:
        print("Not using any cache; starting afresh")
    ds_configs = dataset_utils.get_dataset_info_dicts(ds_name)
    if label_field:
        label_field, label_names = (
            ds_configs[config_name]["features"][label_field][0]
            if len(ds_configs[config_name]["features"][label_field]) > 0
            else ((), [])
        )
    else:
        label_field = ()
        label_names = []
    dataset_args = {
        "dset_name": ds_name,
        "dset_config": config_name,
        "split_name": split_name,
        "text_field": text_field,
        "label_field": label_field,
        "label_names": label_names,
        "calculation": calculation,
        "cache_dir": out_dir,
    }
    load_or_prepare_widgets(dataset_args, use_cache=use_cache)


def main():
    # TODO: Make this the Hugging Face arg parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """

         Example for hate speech18 dataset:
         python3 run_data_measurements.py --dataset="hate_speech18" --config="default" --split="train" --feature="text"

         Example for IMDB dataset:
         python3 run_data_measurements.py --dataset="imdb" --config="plain_text" --split="train" --label_field="label" --feature="text"
         """
        ),
    )

    parser.add_argument(
        "-d", "--dataset", required=True, help="Name of dataset to prepare"
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Dataset configuration to prepare"
    )
    parser.add_argument(
        "-s", "--split", required=True, type=str, help="Dataset split to prepare"
    )
    parser.add_argument(
        "-f",
        "--feature",
        required=True,
        type=str,
        default="text",
        help="Text column to prepare",
    )
    parser.add_argument(
        "-w",
        "--calculation",
        help="""What to calculate (defaults to everything except embeddings).\n
                                                    Options are:\n

                                                    - `general` (for duplicate counts, missing values, length statistics.)\n

                                                    - `lengths` for text length distribution\n

                                                    - `labels` for label distribution\n

                                                    - `embeddings` (Warning: Slow.)\n

                                                    - `npmi` for word associations\n

                                                    - `zipf` for zipfian statistics
                                                    """,
    )
    parser.add_argument(
        "-l",
        "--label_field",
        type=str,
        required=False,
        default="",
        help="Field name for label column in dataset (Required if there is a label field that you want information about)",
    )
    parser.add_argument(
        "--cached",
        default=False,
        required=False,
        action="store_true",
        help="Whether to use cached files (Optional)",
    )
    parser.add_argument(
        "--do_html",
        default=False,
        required=False,
        action="store_true",
        help="Whether to write out corresponding HTML files (Optional)",
    )
    parser.add_argument("--out_dir", default="cache_dir", help="Where to write out to.")
    parser.add_argument(
        "--overwrite_previous",
        default=False,
        required=False,
        action="store_true",
        help="Whether to overwrite a previous cache for these same arguments (Optional)",
    )
    parser.add_argument("--email", default=None, help="Email to report whether the computation was successful")

    args = parser.parse_args()
    print("Proceeding with the following arguments:")
    print(args)
    # run_data_measurements.py -n hate_speech18 -c default -s train -f text -w npmi
    if args.email is not None:
        server = smtplib.SMTP("smtp.gmail.com", port=587)
        server.starttls()
        server.login("data.measurements.tool@gmail.com", EMAIL_PASSWORD)

    dataset_cache_dir = f"{args.dataset}_{args.config}_{args.split}_{args.feature}"
    dataset_arguments_message=f"dataset: {args.dataset}, config: {args.config}, split: {args.split}, feature: {args.feature}, label field: {args.label_field}"
    try:
        create_repo(dataset_cache_dir, organization="datameasurements", repo_type="dataset", private=True, token=HF_TOKEN)
    except Exception as e:
        print(e)
        already_computed_message = f"Already created a repo for the dataset with arguments: {dataset_arguments_message}. Or there is an error on the hub that is preventing repo creation."
        print(already_computed_message)
        if args.overwrite_previous:
            print("Computing new cache regardless.")
        else:
            not_computing_message = "Not computing the dataset cache."
            print(not_computing_message)
            if args.email is not None:
                server.sendmail("data.measurements.tool@gmail.com", args.email, "Subject: Data Measurments not Computed\n\n" + already_computed_message + " " + not_computing_message)
            return
    try:
        cache_path = pjoin(args.out_dir, dataset_cache_dir)
        repo = Repository(local_dir=cache_path, clone_from="datameasurements/" + dataset_cache_dir, repo_type="dataset", use_auth_token=HF_TOKEN)
        repo.lfs_track(["*.feather"])
        get_text_label_df(
            args.dataset,
            args.config,
            args.split,
            args.feature,
            args.label_field,
            args.calculation,
            args.out_dir,
            do_html=args.do_html,
            use_cache=args.cached,
        )
        open(pjoin(cache_path, "computation_result.json"), "w+").write(json.dumps({"complete": True}))
        repo.git_pull()
        repo.push_to_hub(commit_message="Added dataset cache.")

        print()
        if args.email is not None:
            computed_message = f"Data measurements have been computed for dataset with these arguments: {dataset_arguments_message}. You can return to the data measurements tool to view them at https://huggingface.co/spaces/datameasurements/data-measurements-tool"
            server.sendmail("data.measurements.tool@gmail.com", args.email, "Subject: Data Measurements Computed!\n\n" + computed_message)
    except Exception as e:
        print(e)
        if args.email is not None:
            error_message = f"An error occurred in computing data measurements for dataset with arguments: {dataset_arguments_message}. Feel free to make an issue here: https://github.com/huggingface/data-measurements-tool/issues"
            server.sendmail("data.measurements.tool@gmail.com", args.email, "Subject: Data Measurements not Computed\n\n" + error_message)


if __name__ == "__main__":
    main()
