import argparse
import json
from multiprocessing import Process

import loguru
from bs4 import BeautifulSoup
from tqdm import tqdm
import sys
sys.path.append("./")
from html4rag.html_utils import simplify_html


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--rewrite_method", type=str, default="t5")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--keep_attr", action="store_true")
    args = argparser.parse_args()
    rewrite_method = args.rewrite_method
    split = args.split

    # datasets = ["asqa", "hotpot-qa", "nq", "trivia-qa", "musique", "eli5"]
    datasets = ["asqa"]
    # datasets = ["nq"]

    def simplify_html_lines(dataset):
        data_file = f"./html_data/{dataset}/bing/binghtml-{rewrite_method}-{dataset}-{split}.jsonl"
        data_lines = [json.loads(l) for l in open(data_file)]
        if args.keep_attr:
            output_file = f"./html_data/{dataset}/bing/binghtml-{rewrite_method}-{dataset}-simplewithattr-{split}.jsonl"
        else:
            output_file = f"./html_data/{dataset}/bing/binghtml-{rewrite_method}-{dataset}-simple-{split}.jsonl"
        loguru.logger.info(f"Reading data from {data_file}")
        for idx in tqdm(range(len(data_lines)), desc=f"Processing {dataset} {split}"):
            for idj in range(len(data_lines[idx][f'{rewrite_method}_results'])):
                h_soup = BeautifulSoup(data_lines[idx][f'{rewrite_method}_results'][idj]['html'], 'html.parser')
                data_lines[idx][f'{rewrite_method}_results'][idj]['html'] = simplify_html(h_soup, keep_attr=args.keep_attr)

        with open(output_file, "w") as f:
            #  try to encode in utf-8, if it fails, try to replace the characters with ?
            for idl, l in enumerate(data_lines):
                try:
                    f.write(json.dumps(l, ensure_ascii=False) + "\n")
                except Exception as e:
                    loguru.logger.error(f"Line {idl} Error: {e}")
                    f.write(json.dumps(l, ensure_ascii=False).encode("utf-8", errors="replace").decode("utf-8") + "\n")
        loguru.logger.info(f"Saved simplified html to {output_file}")


    child_processes = []
    for dataset in datasets:
        p = Process(target=simplify_html_lines, args=(dataset,))
        child_processes.append(p)
        p.start()

    for p in child_processes:
        p.join()
    loguru.logger.info("All done")
