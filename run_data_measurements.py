import argparse
import json
from dotenv import load_dotenv
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
from os.path import exists, join as pjoin
from pathlib import Path
import utils
from utils import dataset_utils

logs = utils.prepare_logging(__file__)

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


def load_or_prepare(dataset_args, calculation=False, use_cache=False):
    # TODO: Catch error exceptions for each measurement, so that an error
    # for one measurement doesn't break the calculation of all of them.

    do_all = False
    dstats = dataset_statistics.DatasetStatisticsCacheClass(**dataset_args,
                                                            use_cache=use_cache)
    logs.info("Tokenizing dataset.")
    dstats.load_or_prepare_tokenized_df()
    logs.info("Calculating vocab.")
    dstats.load_or_prepare_vocab()

    if not calculation:
        do_all = True

    if do_all or calculation == "general":
        logs.info("\n* Calculating general statistics.")
        dstats.load_or_prepare_general_stats()
        logs.info("Done!")
        logs.info(
            "Basic text statistics now available at %s." % dstats.general_stats_json_fid)

    if do_all or calculation == "duplicates":
        logs.info("\n* Calculating text duplicates.")
        dstats.load_or_prepare_text_duplicates()
        duplicates_fid_dict = dstats.duplicates_files
        logs.info("If all went well, then results are in the following files:")
        for key, value in duplicates_fid_dict.items():
            logs.info("%s: %s" % (key, value))

    if do_all or calculation == "lengths":
        logs.info("\n* Calculating text lengths.")
        dstats.load_or_prepare_text_lengths()
        logs.info("Done!")
        logs.info(
            "- Text length results now available at %s." % dstats.length_df_fid)

    if do_all or calculation == "labels":
        logs.info("\n* Calculating label statistics.")
        dstats.load_or_prepare_labels()
        label_fid_dict = dstats.label_files
        logs.info("If all went well, then results are in the following files:")
        for key, value in label_fid_dict.items():
            logs.info("%s: %s" % (key, value))

    if do_all or calculation == "npmi":
        logs.info("\n* Preparing nPMI.")
        npmi_stats = dataset_statistics.nPMIStatisticsCacheClass(
            dstats, use_cache=use_cache
        )
        do_npmi(npmi_stats)
        logs.info("Done!")
        logs.info(
            "nPMI results now available in %s for all identity terms that "
            "occur more than 10 times and all words that "
            "co-occur with both terms."
            % npmi_stats.pmi_cache_path
        )

    if do_all or calculation == "zipf":
        logs.info("\n* Preparing Zipf.")
        dstats.load_or_prepare_zipf()
        logs.info("Done!")
        zipf_json_fid, zipf_fig_json_fid, zipf_fig_html_fid = zipf.get_zipf_fids(
            dstats.dataset_cache_dir)
        logs.info("Zipf results now available at %s." % zipf_json_fid)
        logs.info(
            "Figure saved to %s, with corresponding json at %s."
            % (zipf_fig_html_fid, zipf_fig_json_fid)
        )

    # Don't do this one until someone specifically asks for it -- takes awhile.
    if calculation == "embeddings":
        logs.info("\n* Preparing text embeddings.")
        dstats.load_or_prepare_embeddings()

    # Don't do this one until someone specifically asks for it -- takes awhile.
    if calculation == "perplexities":
        logs.info("\n* Preparing text perplexities.")
        dstats.load_or_prepare_text_perplexities()


def do_npmi(npmi_stats):
    available_terms = npmi_stats.load_or_prepare_npmi_terms()
    completed_pairs = {}
    logs.info("Iterating through terms for joint npmi.")
    for term1 in available_terms:
        for term2 in available_terms:
            if term1 != term2:
                sorted_terms = tuple(sorted([term1, term2]))
                if sorted_terms not in completed_pairs:
                    term1, term2 = sorted_terms
                    logs.info("Computing nPMI statistics for %s and %s" % (
                        term1, term2))
                    _ = npmi_stats.load_or_prepare_joint_npmi(sorted_terms)
                    completed_pairs[tuple(sorted_terms)] = {}


def pass_args_to_DMT(dset_name, dset_config, split_name, text_field, label_field, label_names, calculation, dataset_cache_dir, prepare_gui=False, use_cache=True):
    if not use_cache:
        logs.info("Not using any cache; starting afresh")
    dataset_args = {
        "dset_name": dset_name,
        "dset_config": dset_config,
        "split_name": split_name,
        "text_field": text_field,
        "label_field": label_field,
        "label_names": label_names,
        "dataset_cache_dir": dataset_cache_dir
    }
    if prepare_gui:
        load_or_prepare_widgets(dataset_args, use_cache=use_cache)
    else:
        load_or_prepare(dataset_args, calculation=calculation, use_cache=use_cache)

def set_defaults(args):
    if not args.config:
        args.config = "default"
        logs.info("Config name not specified. Assuming it's 'default'.")
    if not args.split:
        args.split = "train"
        logs.info("Split name not specified. Assuming it's 'train'.")
    if not args.feature:
        args.feature = "text"
        logs.info("Text column name not given. Assuming it's 'text'.")
    if not args.label_field:
        args.label_field = "label"
        logs.info("Label column name not given. Assuming it's 'label'.")
    return args

def main():
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
        "-c", "--config", required=False, default="", help="Dataset configuration to prepare"
    )
    parser.add_argument(
        "-s", "--split", required=False, default="", type=str,
        help="Dataset split to prepare"
    )
    parser.add_argument(
        "-f",
        "--feature",
        "-t",
        "--text-field",
        required=False,
        nargs="+",
        type=str,
        default="",
        help="Column to prepare (handled as text)",
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
        "--use_cache",
        default=False,
        required=False,
        action="store_true",
        help="Whether to use cached files (Optional)",
    )
    parser.add_argument("--out_dir", default="cache_dir",
                        help="Where to write out to.")
    parser.add_argument(
        "--overwrite_previous",
        default=False,
        required=False,
        action="store_true",
        help="Whether to overwrite a previous local cache for these same arguments (Optional)",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="An email that recieves a message about whether the computation was successful. If email is not None, then you must have EMAIL_PASSWORD=<your email password> for the sender email (data.measurements.tool@gmail.com) in a file named .env at the root of this repo.")
    parser.add_argument(
        "--push_cache_to_hub",
        default=False,
        required=False,
        action="store_true",
        help="Whether to push the cache to an organization on the hub. If you are using this option, you must have HUB_CACHE_ORGANIZATION=<the organization you've set up on the hub to store your cache> and HF_TOKEN=<your hf token> on separate lines in a file named .env at the root of this repo.",
    )
    parser.add_argument("--prepare_GUI_data", default=False, required=False,
                        action="store_true",
                        help="Use this to process all of the stats used in the GUI.")
    parser.add_argument("--keep_local", default=True, required=False,
                        action="store_true",
                        help="Whether to save the data locally.")
    orig_args = parser.parse_args()
    args = set_defaults(orig_args)
    logs.info("Proceeding with the following arguments:")
    logs.info(args)
    # run_data_measurements.py -d hate_speech18 -c default -s train -f text -w npmi
    if args.email is not None:
        if Path(".env").is_file():
            load_dotenv(".env")
        EMAIL_PASSWORD = getenv("EMAIL_PASSWORD")
        context = ssl.create_default_context()
        port = 465
        server = smtplib.SMTP_SSL("smtp.gmail.com", port, context=context)
        server.login("data.measurements.tool@gmail.com", EMAIL_PASSWORD)

    dataset_cache_name, local_dataset_cache_dir = dataset_utils.get_cache_dir_naming(args.out_dir, args.dataset, args.config, args.split, args.feature)
    if not args.use_cache and exists(local_dataset_cache_dir):
        if args.overwrite_previous:
            shutil.rmtree(local_dataset_cache_dir)
        else:
            raise OSError("Cached results for this dataset already exist at %s. "
                          "Delete it or use the --overwrite_previous argument." % local_dataset_cache_dir)

    # Initialize the local cache directory
    dataset_utils.make_path(local_dataset_cache_dir)

    # Initialize the repository
    # TODO: print out local or hub cache directory location.
    if args.push_cache_to_hub:
        repo = dataset_utils.initialize_cache_hub_repo(local_dataset_cache_dir, dataset_cache_name)
    # Run the measurements.
    try:
        pass_args_to_DMT(
            dset_name=args.dataset,
            dset_config=args.config,
            split_name=args.split,
            text_field=args.feature,
            label_field=args.label_field,
            label_names=args.label_names,
            calculation=args.calculation,
            dataset_cache_dir=local_dataset_cache_dir,
            prepare_gui=args.prepare_GUI_data,
            use_cache=args.use_cache,
        )
        if args.push_cache_to_hub:
            repo.push_to_hub(commit_message="Added dataset cache.")
        computed_message = f"Data measurements have been computed for dataset" \
                           f" with these arguments: {args}."
        logs.info(computed_message)
        if args.email is not None:
            computed_message += "\nYou can return to the data measurements tool " \
                                "to view them."
            server.sendmail("data.measurements.tool@gmail.com", args.email,
                            "Subject: Data Measurements Computed!\n\n" + computed_message)
            logs.info(computed_message)
    except Exception as e:
        logs.warning(e)
        error_message = f"An error occurred in computing data measurements " \
                        f"for dataset with arguments: {args}. " \
                        f"Feel free to make an issue here: " \
                        f"https://github.com/huggingface/data-measurements-tool/issues"
        if args.email is not None:
            server.sendmail("data.measurements.tool@gmail.com", args.email,
                            "Subject: Data Measurements not Computed\n\n" + error_message)
        logs.warning("Data measurements not computed. ☹️")
        logs.warning(error_message)
        return
    if not args.keep_local:
        # Remove the dataset from local storage - we only want it stored on the hub.
        logs.warning("Deleting measurements data locally at %s" % local_dataset_cache_dir)
        shutil.rmtree(local_dataset_cache_dir)
    else:
        logs.info("Measurements made available locally at %s" % local_dataset_cache_dir)


if __name__ == "__main__":
    main()
