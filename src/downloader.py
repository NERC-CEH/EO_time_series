# encoding: utf-8
"""

From CEDA github with madditions and changes by Ciaran Robb

===================


"""

# Import standard libraries
import os
import sys
import datetime
import requests
import warnings
# Import third-party libraries
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from contrail.security.onlineca.client import OnlineCaClient
from joblib import Parallel, delayed
from tqdm import tqdm

CERTS_DIR = os.path.expanduser('~/.certs')
if not os.path.isdir(CERTS_DIR):
    os.makedirs(CERTS_DIR)

TRUSTROOTS_DIR = os.path.join(CERTS_DIR, 'ca-trustroots')
CREDENTIALS_FILE_PATH = os.path.join(CERTS_DIR, 'credentials.pem')

TRUSTROOTS_SERVICE = 'https://slcs.ceda.ac.uk/onlineca/trustroots/'
CERT_SERVICE = 'https://slcs.ceda.ac.uk/onlineca/certificate/'


def cert_is_valid(cert_file, min_lifetime=0):
    """
    Returns boolean - True if the certificate is in date.
    Optional argument min_lifetime is the number of seconds
    which must remain.
    :param cert_file: certificate file path.
    :param min_lifetime: minimum lifetime (seconds)
    :return: boolean
    """
    try:
        with open(cert_file, 'rb') as f:
            crt_data = f.read()
    except IOError:
        return False

    try:
        cert = x509.load_pem_x509_certificate(crt_data, default_backend())
    except ValueError:
        return False

    now = datetime.datetime.now()

    return (cert.not_valid_before <= now
            and cert.not_valid_after > now + datetime.timedelta(0, min_lifetime))


def setup_credentials():
    """
    Download and create required credentials files.
    Return True if credentials were set up.
    Return False is credentials were already set up.
    :param force: boolean
    :return: boolean
    """

    # Test for DODS_FILE and only re-get credentials if it doesn't
    # exist AND `force` is True AND certificate is in-date.
    if cert_is_valid(CREDENTIALS_FILE_PATH):
        print('[INFO] Security credentials already set up.')
        return False

    
    username = os.environ['CEDA_USERNAME']
    password = os.environ['CEDA_PASSWORD']

    onlineca_client = OnlineCaClient()
    onlineca_client.ca_cert_dir = TRUSTROOTS_DIR

    # Set up trust roots
    trustroots = onlineca_client.get_trustroots(
        TRUSTROOTS_SERVICE,
        bootstrap=True,
        write_to_ca_cert_dir=True)

    # Write certificate credentials file
    key_pair, certs = onlineca_client.get_certificate(
        username,
        password,
        CERT_SERVICE,
        pem_out_filepath=CREDENTIALS_FILE_PATH)

    print('[INFO] Security credentials set up.')
    return True

def setup_sesh(user, password):
    
    """
    setup user/pass
    
    Parameters
    ----------
    
    user: string
        CEDA username
    
    password: string
        CEAD password

    """
    os.environ['CEDA_USERNAME'] = user
    os.environ['CEDA_PASSWORD'] = password


def dload(file_url, folder):
    """
    Download a file from the CEDA archive
    
    Parameters
    ----------
    
    file_url: string
        URL to a NetCDF4 opendap end-point.
    
    folder: string
        the dir in which to download the file
        
    Returns
    -------
    path of file

    """

    try:
        setup_credentials()
    except KeyError:
        print("CEDA_USERNAME and CEDA_PASSWORD environment variables required")
        return

    # Download file to current working directory
    response = requests.get(file_url, cert=(CREDENTIALS_FILE_PATH), verify=False)
    filename = file_url.rsplit('/', 1)[-1]
    final = os.path.join(folder, filename)
    with open(final, 'wb') as file_object:
        file_object.write(response.content)
    
    return final


def dloadbatch(urls, folder, para=False, nt=-1):
    
    """
    Download a batch of files from the CEDA archive
    
    Parameters
    ----------
    
    urls: string
        URLs to a NetCDF4s opendap end-point.
    
    folder: string
        the dir in which to download the file
        
    para: bool
        whether to process in parallel def. False
    
    Returns
    -------
    list of file paths
    """

    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if para == True:
                paths = Parallel(n_jobs=nt, verbose=2)(delayed(dload)(u, folder) for u in urls)
            else:
                paths = [dload(u, folder) for u in tqdm(urls)]
    return paths
    


