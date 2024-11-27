import argparse
import base64
import concurrent.futures
import os
import threading
from typing import Optional

import loguru
import requests
from tqdm import tqdm
import json


def parse_url(
        url: str,
        service_url: Optional[str] = None,
        need_links: bool = False,
        enable_render: bool = False,
        extract_level: int = 1,
) -> str:
    """
    Parse url into html text or raw text.
    Args:
        url (str): url to parse
        service_url (str, optional): service url. Defaults to None.
        enable_render (bool, optional): enable render on url to parse. Defaults to True.
        extract_level (int, optional): extract level. Defaults to 1. Here are the options:
            0: extract raw text only.  1: Just remove js, css, etc on url to parse.
            2: keep all. return raw html text. Note that 2, 3 will return base64 encoded html text.
    """
    if service_url is None:
        # service_url = "http://172.16.100.225:8081/fetch"  # online
        # service_url = "http://lb-2nshjbik-dfpxqkgc3sr5jomn.clb.ap-guangzhou.tencentclb.com/fetch"
        # service_url = "http://172.16.98.152:8081/fetch"  # offline
        service_url = "http://lb-3elxe8bu-3vp6fdo91l25hau7.clb.ap-guangzhou.tencentclb.com/fetch"
    payload = json.dumps({
        "url": url,
        "need_links": need_links,
        "enable_render": enable_render,
        "extract_level": extract_level,
    })
    # print(f"parse url: {url}")
    headers = {'Content-Type': 'application/json'}
    response = requests.request("GET", service_url, headers=headers, data=payload)
    if response.status_code != 200:
        raise Exception(f"Fail to request url: {url}, code: {response.status_code}")
    response_json = json.loads(response.text)
    # if response_json["status_code"] != 0:
    #     raise Exception(f"parse url {url} failed")
    parsed_html = response_json["content"]
    if extract_level != 0:
        try:
            parsed_html = base64.b64decode(parsed_html).decode("utf-8")
        except Exception:
            loguru.logger.warning(f"error decoding html from {url}")
            pass
    return parsed_html


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="wstqa")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    args = argparser.parse_args()
    dataset = args.dataset
    split = args.split
    rewrite_method = args.rewrite_method

    if rewrite_method == "vanilla_search":
        file_path = f"./html_data/{dataset}/bing/bing-{dataset}-{split}.jsonl"
    else:
        file_path = f"./html_data/{dataset}/bing/bing-{rewrite_method}-{dataset}-{split}.jsonl"
    data_lines = [json.loads(l) for l in open(file_path)]

    output_dir = f"./html_data/{dataset}/bing"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/binghtml-{rewrite_method}-{dataset}-{split}.jsonl"

    with open(output_path, "w") as f:
        pass

    if args.mini_dataset:
        data_lines = data_lines[:10]
    loguru.logger.info(f"dataset {dataset}, split {split}, len {len(data_lines)}")
    empty_html = 0

    # with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
    max_workers = 32
    for idx in tqdm(range(len(data_lines))):
        #  send request to get html content
        unique_urls = set()
        for lidx, link in enumerate(data_lines[idx][f'{rewrite_method}_results']):
            # concat_url = link['url']
            unique_urls.add(link['url'])
        data_lines[idx][f'{rewrite_method}_results'] = []
        unique_urls = list(unique_urls)
        htmls = ["" for _ in range(len(unique_urls))]
        threads = []

        global valid_html

        def get_html(i):
            url = unique_urls[i]
            try:
                non_empty_html = len([h for h in htmls if h])
                if non_empty_html >= 50:
                    return
                htmls[i] = parse_url(url, None, False, False, 2)
            except:
                loguru.logger.warning(f"error requesting {url}")
                htmls[i] = ""


        for uidx, url in enumerate(unique_urls):
            t = threading.Thread(target=get_html, args=(uidx,))
            threads.append(t)
            t.start()
            if len(threads) >= max_workers:
                for t in threads:
                    t.join()
                threads = []

        if len(threads) > 0:
            for t in threads:
                t.join()

        for uidx, url in enumerate(unique_urls):
            html = htmls[uidx]
            if not html:
                empty_html += 1
            else:
                data_lines[idx][f'{rewrite_method}_results'].append({
                    "url": url,
                    "html": html
                })
                #  stop if non-empty html is enough
                if len(data_lines[idx][f'{rewrite_method}_results']) >= 50:
                    break

        with open(output_path, "a") as f:
            f.write(json.dumps(data_lines[idx], ensure_ascii=False) + "\n")

    loguru.logger.info(f"saved to {output_path}")
