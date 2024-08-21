"""
Author: Dr. Newton H Nguyen

This is a script to download CMIP data from the ESGF Node
The default node is Lawrence Livermore National Laboratory (LLNL)
The script uses the pyesgf library to search for data on the ESGF Node
The script then uses the wget command to download the data
The script requires a JSON file containing the search parameters

Run example:
python download_cmip.py cesm.json

The JSON file should have the following format:
{
    "project": "CMIP6",
    "experiment": "historical",
    "variables": ["tas"],
    "frequency": "mon",
    "model": "CESM2",
    "realm": "land",
    "ensemble": "r8i1p1f1",
    "start_date": "1950-01-01",
    "end_date": "1999-12-31",
    "latest": true,
    "output_dir": "/path/to/output/directory"
}

All Rights Reserved.
"""

import json
import os
import subprocess
from pyesgf.search import SearchConnection
from pyesgf.logon import LogonManager
import itertools
import requests
import argparse
import warnings
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Turn off warnings
warnings.filterwarnings("ignore")

def load_params(file_path):
    """
    Load the search parameters from a JSON file

    Args:
    file_path: path to the JSON file containing the search parameters

    Returns:
    dictionary containing the search parameters
    """

    with open(file_path, 'r') as file:
        return json.load(file)


def _create_combinations(params):
    """
    Create all possible combinations of the search parameters

    Args:
    params: dictionary containing the search parameters

    Returns:
    list of tuples, where each tuple contains the search parameters for a single search
    """

    project = params.get('project', 'CMIP6')
    experiment = params.get('experiment', 'historical')
    variables = params.get('variables', [])
    frequency = params.get('frequency', 'mon')
    model = params.get('model', 'CESM2')
    realm = params.get('realm', 'land')
    ensemble = params.get('ensemble', 'r8i1p1f1')
    start_date = params.get('start_date', '1950-01-01')
    end_date = params.get('end_date', '1999-12-31')
    latest = params.get('latest', True)

    return itertools.product(
        project if isinstance(project, list) else [project],
        experiment if isinstance(experiment, list) else [experiment],
        variables if isinstance(variables, list) else [variables],
        frequency if isinstance(frequency, list) else [frequency],
        model if isinstance(model, list) else [model],
        realm if isinstance(realm, list) else [realm],
        ensemble if isinstance(ensemble, list) else [ensemble],
        start_date if isinstance(start_date, list) else [start_date],
        end_date if isinstance(end_date, list) else [end_date],
        [latest]
    )


def _perform_search(conn, combo):
    """
    Perform a search on ESGF using the given search parameters
    
    Args:
    conn: SearchConnection object
    combo: tuple containing the search parameters
    
    Returns:
    files_info: list of dictionaries containing information about the files found
    total_size: total size of all files found
    ctx: SearchContext object
    """

    project, experiment, variable, frequency, model, realm, ensemble, start_date, end_date, latest = combo
    
    facets = {
        'project': project,
        'experiment_id': experiment,
        'variable_id': variable,
        'frequency': frequency,
        'source_id': model,
        'realm': realm,
        'member_id': ensemble,
        'start': start_date,
        'end': end_date
    }

    facets_str = ','.join(f'{key}={value}' for key, value in facets.items())

    ctx = conn.new_context(
        project=project,
        experiment_id=experiment,
        variable_id=variable,
        frequency=frequency,
        source_id=model,
        realm=realm,
        member_id=ensemble,
        start=start_date,
        end=end_date,
        latest=latest,
        facets=facets_str
    )
    
    files_info = []
    total_size = 0
    for result in ctx.search(ignore_facet_check=True):
        for file in result.file_context().search(ignore_facet_check=True):
            files_info.append({'url': file.download_url, 'filename': file.filename, 'size': file.size})
            total_size += file.size

    return files_info, total_size, ctx


def _download_files(ctx_searches, output_dir):
    """
    Download the files found in the search results

    Args:
    ctx_searches: list of SearchContext objects
    output_dir: directory where the files will be saved
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for ctx in ctx_searches:
        for result in ctx.search(ignore_facet_check=True):
            try:
                # Fetch the download script for the dataset
                fc = result.file_context()
                wget_script_content = fc.get_download_script()

                # Save the wget script to a temporary file
                script_path = tempfile.mkstemp(suffix='.sh', prefix='download-')[1]
                with open(script_path, "w") as writer:
                    writer.write(wget_script_content)

                # Make the script executable
                os.chmod(script_path, 0o750)

                # Run the script to download the files
                message = subprocess.run([script_path, '-s'], cwd=output_dir, capture_output=True, text=True)

                # error checking
                if "206 Partial Content" in message.stdout or "Requested Range Not Satisfiable" in message.stdout:
                    logging.info(f"Skipping file as it's already fully retrieved or unsuitable: {script_path}")
                    continue


                logging.info(f"Downloaded files for dataset {result.dataset_id} successfully.")
            except Exception as e:
                logging.error(f"Failed to download files for dataset {result.dataset_id}: {e}")

    logging.info("All downloads complete.")
    return


def download_cmip6_data(params, output_dir):
    """
    Download CMIP6 data from ESGF
    
    Args:
    params: dictionary containing the search parameters
    output_dir: directory where the files will be saved
    
    Returns:
    None
    """

    # Set up ESGF credentials
    #lm = LogonManager()
    
    # Uncomment the method you want to use
    # Method 1: Use Username and Password
    #lm.logon(username=None, password=None, hostname='esgf-node.llnl.gov', bootstrap=True, interactive=True)
    
    # Method 2: Use OpenID
    # lm.logon_with_openid(openid               ='https://esgf-node.llnl.gov/esgf-idp/openid/your_openid', password='your_password', bootstrap=True)
    
    #if not lm.is_logged_on():
        #print("Failed to log in to ESGF.")
        #return

    conn = SearchConnection('https://esgf-node.llnl.gov/esg-search', distrib=True)
    combinations = _create_combinations(params)

    all_files_info = []
    total_size = 0
    all_ctx = []

    for combo in combinations:
        files_info, combo_size, search_ctx = _perform_search(conn, combo)
        all_files_info.extend(files_info)
        total_size += combo_size
        all_ctx.append(search_ctx)

    print("Total size of all downloads: {:.2f} MB".format(total_size / (1024 * 1024)))
    print("Number of files to download: {}".format(len(all_files_info)))
    
    confirm = input("Do you want to proceed with the download? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Download aborted by user.")
        return

    # Download files
    _download_files(all_ctx, output_dir)

    print("Download complete!")
    print("saved to {}".format(output_dir))
    return

# read the parameters from the commandline
# parameters are in a jason file
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download CMIP6 data')
    parser.add_argument('params_file', help='Path to the JSON file containing the search parameters')
    args = parser.parse_args()

    params = load_params(args.params_file)
    output_dir = params.get('output_dir', os.getcwd())
    download_cmip6_data(params, output_dir=output_dir)