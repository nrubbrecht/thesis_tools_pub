import requests
import os
import json
import re


def sanitize_filename(filename):
    # Replace invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def query_pds_api(product_id):
    base_url = "https://pds-imaging.jpl.nasa.gov/solr/pds_archives/search"
    params = {
        'q': f'identifier:1_{product_id}*',  # Using the 'identifier' as the query parameter
        'wt': 'json',
        'rows': 10  # Fetch up to 10 records (adjust as needed)
    }
    try:
        response = requests.get(base_url, params=params, verify=False)  # Disable SSL verification for testing
        response.raise_for_status()  # Raise an error for bad responses

        # Print raw response for debugging purposes
        # print("Raw API response:")
        # print(json.dumps(response.json(), indent=4))  # Pretty print the JSON response

        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    return None


def download_file(url, download_dir, filename):
    file_path = os.path.join(download_dir, filename)
    if not os.path.exists(file_path):
        try:
            response = requests.get(url, verify=False)  # Disable SSL verification for testing
            response.raise_for_status()

            with open(file_path, 'wb') as file:
                file.write(response.content)

            print(f'Downloaded {url} to {file_path}')
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")
    else:
        print(f"File {file_path} already exists. Skipping download.")


def main_search(product_id, download_dir):
    data = query_pds_api(product_id)
    browse_url = "None found"
    if data and data['response']['numFound'] > 0:
        doc = data['response']['docs'][0]  # Assuming we're interested in the first document

        # Extract the URLs from the JSON response
        browse_url = doc.get('ATLAS_BROWSE_URL')
        label_url = doc.get('ATLAS_LABEL_URL')

        if browse_url:
            print(f"Browse Image URL: {browse_url}")
            # Create a filename for the browse image
            browse_filename = f"{sanitize_filename(product_id)}.jpeg"
            # Download browse image
            download_file(browse_url, download_dir, browse_filename)

        if label_url:
            print(f"Label File URL: {label_url}")
            # Create a filename for the label file
            label_filename = f"{sanitize_filename(product_id)}.lbl"
            # Download label file
            download_file(label_url, download_dir, label_filename)

    else:
        print("No results found for the given product ID.")

    return browse_url

if __name__ == "__main__":
    product_id = "N1711553692"  # Replace with your specific product ID or use wildcard
    download_dir = "downloaded_files"  # Directory to save the downloaded files
    os.makedirs(download_dir, exist_ok=True)  # Ensure the directory exists
    main_search(product_id, download_dir)
