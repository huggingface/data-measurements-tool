import argparse
import json
# TODO(Tristan): Fix this dependency
# from dotenv import load_dotenv
import plotly
import shutil
import smtplib
import ssl
import sys
import textwrap
from data_measurements import dataset_statistics
from data_measurements.zipf import zipf
from huggingface_hub import create_repo, Repository, hf_api
from os import getenv
from os.path import join as pjoin
from pathlib import Path
from utils import dataset_utils

port = 465  # For SSL

# if Path(".env").is_file():
#    load_dotenv(".env")

# TODO: Explain that this needs to be done/how to do it.
HF_TOKEN = getenv("HF_TOKEN")
EMAIL_PASSWORD = getenv("EMAIL_PASSWORD")


def load_or_prepare_widgets(ds_args, show_embeddings=False,
                            show_perplexities=False, use_cache=False):
    """
    Loader specifically for the widgets used in the app.
    Args:
        ds_args:
        show_embeddings:
        show_perplexities:
        use_cache:

    Returns:

    """
    dataset_utils.make_path(ds_args["cache_dir"])
    dstats = dataset_statistics.DatasetStatisticsCacheClass(**ds_args,
                                                            use_cache=use_cache)
    # Header widget
    dstats.load_or_prepare_dset_peek()
    # General stats widget
    dstats.load_or_prepare_general_stats()
    # Labels widget
    dstats.load_or_prepare_labels()
    # Text lengths widget
    dstats.load_or_prepare_text_lengths()
    if show_embeddings:
        # Embeddings widget
        dstats.load_or_prepare_embeddings()
    if show_perplexities:
        # Text perplexities widget
        dstats.load_or_prepare_text_perplexities()
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
    # TODO: Catch error exceptions for each measurement, so that an error
    # for one measurement doesn't break the calculation of all of them.

    do_all = False
    print(dataset_args)
    dstats = dataset_statistics.DatasetStatisticsCacheClass(**dataset_args,
                                                            use_cache=use_cache)
    print("Loading dataset.")
    dstats.load_or_prepare_dataset()
    print("Dataset loaded.  Preparing vocab.")
    dstats.load_or_prepare_tokenized_df()
    print("Tokenized.")
    dstats.load_or_prepare_vocab()
    print("Vocab prepared.")

    if not dataset_args["calculation"]:
        do_all = True

    if do_all or dataset_args["calculation"] == "general":
        print("\n* Calculating general statistics.")
        dstats.load_or_prepare_general_stats()
        print("Done!")
        print(
            "Basic text statistics now available at %s." % dstats.general_stats_json_fid)

    if do_all or dataset_args["calculation"] == "duplicates":
        print("\n* Calculating text duplicates.")
        dstats.load_or_prepare_text_duplicates()
        duplicates_fid_dict = dstats.duplicates_files
        print("If all went well, then results are in the following files:")
        for key, value in duplicates_fid_dict.items():
            print("%s: %s" % (key, value))
        print()

    if do_all or dataset_args["calculation"] == "lengths":
        print("\n* Calculating text lengths.")
        dstats.load_or_prepare_text_lengths()
        print("Done!")
        print(
            "- Text length results now available at %s." % dstats.length_df_fid)
        print()

    if do_all or dataset_args["calculation"] == "labels":
        print("\n* Calculating label statistics.")
        dstats.load_or_prepare_labels()
        label_fid_dict = dstats.label_files
        print("If all went well, then results are in the following files:")
        for key, value in label_fid_dict.items():
            print("%s: %s" % (key, value))
        print()



    if do_all or dataset_args["calculation"] == "npmi":
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

    if do_all or dataset_args["calculation"] == "zipf":
        print("\n* Preparing Zipf.")
        dstats.load_or_prepare_zipf()
        print("Done!")
        zipf_json_fid, zipf_fig_json_fid, zipf_fig_html_fid = zipf.get_zipf_fids(
            dstats.cache_path)
        print("Zipf results now available at %s." % zipf_json_fid)
        print(
            "Figure saved to %s, with corresponding json at %s."
            % (zipf_fig_html_fid, zipf_fig_json_fid)
        )

    # Don't do this one until someone specifically asks for it -- takes awhile.
    if dataset_args["calculation"] == "embeddings":
        print("\n* Preparing text embeddings.")
        dstats.load_or_prepare_embeddings()

    # Don't do this one until someone specifically asks for it -- takes awhile.
    if dataset_args["calculation"] == "perplexities":
        print("\n* Preparing text perplexities.")
        dstats.load_or_prepare_text_perplexities()


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
                    print("Computing nPMI statistics for %s and %s" % (
                        term1, term2))
                    _ = npmi_stats.load_or_prepare_joint_npmi(sorted_terms)
                    completed_pairs[tuple(sorted_terms)] = {}


def get_text_label_df(
        ds_name,
        config_name,
        split_name,
        text_field,
        label_field,
        label_names,
        calculation,
        out_dir,
        do_html=False,
        prepare_gui=False,
        use_cache=True,
):
    if not use_cache:
        print("Not using any cache; starting afresh")

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
    if prepare_gui:
        load_or_prepare_widgets(dataset_args, use_cache=use_cache)
    else:
        load_or_prepare(dataset_args, use_cache=use_cache)


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
        "-s", "--split", required=True, type=str,
        help="Dataset split to prepare"
    )
    parser.add_argument(
        "-f",
        "--feature",
        required=True,
        nargs="+",
        type=str,
        default="text",
        help="Text column to prepare",
    )
    parser.add_argument(
        "-w",
        "--calculation",
        help="""What to calculate (defaults to everything except embeddings and perplexities).\n
                                                    Options are:\n

                                                    - `general` (for duplicate counts, missing values, length statistics.)\n

                                                    - `duplicates` for duplicate counts\n

                                                    - `lengths` for text length distribution\n

                                                    - `labels` for label distribution\n

                                                    - `embeddings` (Warning: Slow.)\n

                                                    - `perplexities` (Warning: Slow.)\n

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
    parser.add_argument('-n', '--label_names', nargs='+', default=[])
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
    parser.add_argument("--out_dir", default="cache_dir",
                        help="Where to write out to.")
    parser.add_argument(
        "--overwrite_previous",
        default=False,
        required=False,
        action="store_true",
        help="Whether to overwrite a previous cache for these same arguments (Optional)",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="An email that recieves a message about whether the computation was successful. If email is not None, then you must have EMAIL_PASSWORD for the sender email (data.measurements.tool@gmail.com) in a file named .env at the root of this repo.")
    parser.add_argument(
        "--push_cache_to_hub",
        default=False,
        required=False,
        action="store_true",
        help="Whether to push the cache to the datameasurements organization on the hub. If you are using this option, you must have HF_TOKEN in a file named .env at the root of this repo.",
    )
    parser.add_argument("--prepare_GUI_data", default=False, required=False,
                        action="store_true",
                        help="Use this to process all of the stats used in the GUI.")
    parser.add_argument("--keep_local", default=True, required=False,
                        action="store_true",
                        help="Whether to save the data locally.")

    args = parser.parse_args()
    print("Proceeding with the following arguments:")
    print(args)
    # run_data_measurements.py -d hate_speech18 -c default -s train -f text -w npmi
    if args.email is not None:
        context = ssl.create_default_context()
        server = smtplib.SMTP_SSL("smtp.gmail.com", port, context=context)
        server.login("data.measurements.tool@gmail.com", EMAIL_PASSWORD)

    # The args specify that multiple features can be selected.
    # We combine them for the filename here.
    args.feature = ".".join(args.feature)

    dataset_cache_dir = f"{args.dataset}_{args.config}_{args.split}_{args.feature}"
    cache_path = args.out_dir + "/" + dataset_cache_dir
    dataset_utils.make_path(cache_path)

    dataset_arguments_message = f"dataset: {args.dataset}, config: {args.config}, split: {args.split}, feature: {args.feature}, label field: {args.label_field}, label names: {args.label_names}"
    # Prepare some of the messages we use in different if-else/try-except cases later.
    not_computing_message = "As you specified, not overwriting the previous dataset cache."
    # Initialize the repository
    if args.push_cache_to_hub:
        try:
            create_repo(dataset_cache_dir, organization="datameasurements",
                        repo_type="dataset", private=True, token=HF_TOKEN)
        # Error because the repo is already created
        except hf_api.HTTPError as err:
            if err.args[
                0] == "409 Client Error: Conflict for url: https://huggingface.co/api/repos/create - You already created this dataset repo":
                already_computed_message = f"Already created a repo for the dataset with arguments: {dataset_arguments_message}."
            else:
                already_computed_message = " - ".join(err.args)
            print(already_computed_message)
            if args.overwrite_previous:
                print("As you specified, overwriting previous cache.")
            else:
                print(not_computing_message)
                if args.email is not None:
                    server.sendmail("data.measurements.tool@gmail.com",
                                    args.email,
                                    "Subject: Data Measurments not Computed\n\n" + already_computed_message + " " + not_computing_message)
                return
        # Some other error that we do not anticipate.
        except Exception as err:
            error_message = f"There is an error on the hub that is preventing repo creation. Details: " + " - ".join(
                err.args)
            print(error_message)
            print(not_computing_message)
            if args.email is not None:
                server.sendmail("data.measurements.tool@gmail.com", args.email,
                                "Subject: Data Measurments not Computed\n\n" + error_message + "\n" + not_computing_message)
            return
    # Run the measurements.
    try:
        if args.push_cache_to_hub:
            # TODO: This breaks if the cache path exists and it isn't a git repository (such as when someone has dev'ed locally and is moving to the online repo)
            # A solution would be something like this, although probably the .git directory
            # would also need to be checked to see if it's the *right* git repo.
            """
            n = 1
            new_cache_path = cache_path
            while os.path.exists(new_cache_path) and not os.path.exists(new_cache_path + "/.git"):
               print("Trying to clone from repo to %s, but it exists already and is not a git repo." % new_cache_path)
               new_cache_path = cache_path + "." + str(n)
               print("Trying to clone to %s instead" % new_cache_path)
               n += 1
            """
            repo = Repository(local_dir=cache_path,
                              clone_from="datameasurements/" + dataset_cache_dir,
                              repo_type="dataset", use_auth_token=HF_TOKEN)
            repo.lfs_track(["*.feather"])
        get_text_label_df(
            args.dataset,
            args.config,
            args.split,
            args.feature,
            args.label_field,
            args.label_names,
            args.calculation,
            args.out_dir,
            do_html=args.do_html,
            prepare_gui=args.prepare_GUI_data,
            use_cache=args.cached,
        )
        if args.push_cache_to_hub:
            repo.push_to_hub(commit_message="Added dataset cache.")
        computed_message = f"Data measurements have been computed for dataset with these arguments: {dataset_arguments_message}."
        print(computed_message)
        print()
        if args.email is not None:
            computed_message += "\nYou can return to the data measurements tool to view them at https://huggingface.co/spaces/datameasurements/data-measurements-tool"
            server.sendmail("data.measurements.tool@gmail.com", args.email,
                            "Subject: Data Measurements Computed!\n\n" + computed_message)
            print(computed_message)
    except Exception as e:
        print(e)
        error_message = f"An error occurred in computing data measurements for dataset with arguments: {dataset_arguments_message}. Feel free to make an issue here: https://github.com/huggingface/data-measurements-tool/issues"
        if args.email is not None:
            server.sendmail("data.measurements.tool@gmail.com", args.email,
                            "Subject: Data Measurements not Computed\n\n" + error_message)
        print()
        print("Data measurements not computed. ☹️")
        print(error_message)
        return
    if not args.keep_local:
        # Remove the dataset from local storage - we only want it stored on the hub.
        print("Deleting measurements data locally at %s" % cache_path)
        shutil.rmtree(cache_path)
    else:
        print("Measurements made available locally at %s" % cache_path)


if __name__ == "__main__":
    main()

    # Deleted this because of merge conflict -- saving here in case it should not have been deleted.
    # try:
    #    create_repo(dataset_cache_dir, organization="datameasurements", repo_type="dataset", private=True, token=HF_TOKEN)
    # except hf_api.HTTPError as err:
    #    if err.args[0] == "409 Client Error: Conflict for url: https://huggingface.co/api/repos/create - You already created this dataset repo":
    #        error_message = f"Already created a repo for the dataset with arguments: {dataset_arguments_message}."
    #    else:
    #        error_message = " - ".join(err.args)
    #    print(error_message)
    #    if args.overwrite_previous:
    #        print("Overwriting precious cache.")
    ## May never hit this.
    # except Exception as err:
    #    error_message = f"There is an error on the hub that is preventing repo creation. Details: " + " - ".join(err.args)
    #    not_computing_message = "Not computing the dataset cache."
    #    print(error_message)
    #    print(not_computing_message)
    #    if args.email is not None:
    #        server.sendmail("data.measurements.tool@gmail.com", args.email, "Subject: Data Measurments not Computed\n\n" + error_message + "\n" + not_computing_message)
    #    return
    # try:
    #    cache_path = args.out_dir + "/" + dataset_cache_dir
    #    repo = Repository(local_dir=cache_path, clone_from="datameasurements/" + dataset_cache_dir, repo_type="dataset", use_auth_token=HF_TOKEN)
    #    repo.lfs_track(["*.feather"])
