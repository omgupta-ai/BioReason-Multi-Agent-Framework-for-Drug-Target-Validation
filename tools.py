from Bio import Entrez
from chembl_webresource_client.new_client import new_client
from langchain.tools import tool

Entrez.email = "gupta.om@northeastern.edu"

@tool
def pubmed_search(query:str):
    """
    Search PubMed for medical literature. 
    Use this when you need to find research papers or abstracts about a drug target.
    """
    Entrez.email = "your-email@northeastern.edu"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=3)
    record = Entrez.read(handle)
    id_list = record["IdList"]
    if not id_list: return "No papers found."
    fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="text")
    return fetch_handle.read()

@tool
def chembl_search(target_name:str):
    """
    Search ChEMBL for molecular bioactivity data.
    Use this to find IC50 values and binding information for a specific protein target.
    """
    target = new_client.target
    res = target.filter(pref_name__icontains=target_name).only('target_chembl_id')
    if not res: return "Target not found."
    
    activities = new_client.activity.filter(target_chembl_id=res[0]['target_chembl_id']).filter(standard_type="IC50")[:3]
    return str(activities)
