# https://rdflib.github.io/sparqlwrapper/

import sys
import time
from SPARQLWrapper import SPARQLWrapper, JSON, CSV

# %%

endpoint_url = "https://query.wikidata.org/sparql"

# this query queries 100 humans that are female

query_human_females_100 = """SELECT ?item ?itemLabel WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
  ?item wdt:P21 wd:Q6581072;
    wdt:P31 wd:Q5.
}
LIMIT 100"""

query_human_females = """SELECT ?item ?itemLabel WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
  ?item wdt:P21 wd:Q6581072;
    wdt:P31 wd:Q5.
}
LIMIT 1000"""

def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

# %% Query Wikidata

# select a query

query = query_human_females_100

print('Query has started.')
start_time = time.perf_counter()
results = get_results(endpoint_url, query)
end_time = time.perf_counter()

print(f'This query took {round(end_time-start_time, 2)} seconds.')  # 44.38


# %% Look at the results

# looking at results in JSON format
# result is a dict of dicts, here: 1 dict for item, 1 for itemLabel (refer to query)
for result in results["results"]["bindings"]:
    print(result.get('itemLabel').get('value'))



# transform to dataframe
import pandas as pd

# THIS WORKS FOR RE-ROFRMATTING NORMAL SELECT QUERIES
# results.get('results').get('bindings')  # is a list of dicts
results_df = pd.json_normalize(results.get('results').get('bindings'))

# THIS WORKS FOR RE-ROFRMATTING COUNT QUERIES


# result is a nice dataframe

# %% REQUESTS LIBRARY
# do a query using the requests library

import requests
import pandas as pd

wikidata_URL = "https://query.wikidata.org/sparql"

query_EU_countries = """
SELECT ?item ?itemLabel WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
  ?item wdt:P463 wd:Q458.
}
"""

result_requests = requests.get(wikidata_URL, params = {'format': 'json',
                                                       'query': query_EU_countries})
data = result_requests.json()

temp = pd.json_normalize(data['results']['bindings'])  # index JSON file to get list of dicts


# %% QUERYING FOR SENSITIVE ATTRIBUTES

query_gender_sexualorien = """
SELECT ?item ?itemLabel ?gender ?sexualOrientation
WHERE {
  ?item wdt:P21 ?gender.
  ?item wdt:P91 ?sexualOrientation.
  ?item wdt:P31 wd:Q5.
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}  
"""

result = requests.get(wikidata_URL, params = {'query': query_gender_sexualorien,
                                              'format': 'json'})
data = result.json()
data_df = pd.json_normalize(data['results']['bindings'])
