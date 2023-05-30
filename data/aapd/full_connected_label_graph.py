import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", help="number of labels", default=50)
    parser.add_argument("--h", help="the size of hidden state", default=768)
    parser.add_argument("--output", default="full_connected_label_graph_bert50")
    return parser.parse_args()


def main(args):
    os.makedirs(args.output, exist_ok=True)

    # write nodes.csv
    with open(os.path.join(args.output, "nodes.csv"), "w") as fh:
        fh.write("node_id,feature\n")
        for index in range(args.n):
            feature = '"' + ",".join([
                str(item) for item in np.random.random(args.h).tolist()
            ]) + '"'
            fh.write(f"{index},{feature}\n")

    # write edges.csv
    with open(os.path.join(args.output, "edges.csv"), "w") as fh:
        fh.write("src_id,dst_id\n")
        for src in range(args.n):
            for dst in range(src + 1, args.n):
                fh.write(f"{src},{dst}\n")

        # meta.yaml
    with open(os.path.join(args.output, "meta.yaml"), "w") as fh:
        fh.write("\n".join([
            f"dataset_name: {args.output}",
            "node_data:",
            "- file_name: nodes.csv",
            "  ntype: item",
            "edge_data:",
            "- file_name: edges.csv",
            "  etype: [item, edge, item]",
        ]))


if __name__ == '__main__':
    main(parse_args())
