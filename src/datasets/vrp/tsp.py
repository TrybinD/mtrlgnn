from src.datasets.base import BaseDataset
from src.problems.vrp.tsp import TSP, TSPInstance, TSPSolution
from src.utils import normalize_coord, dist_to_coords
import torch
import requests
import tarfile
import gzip
import shutil
import os
from tsplib95.loaders import load


def download_tsplib95():
    url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz"
    extract_to = "tsplib95_data"
    local_filename = url.split("/")[-1]

    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the tar.gz file
    with tarfile.open(local_filename, "r:gz") as tar:
        tar.extractall(path=extract_to)

    # Extract all nested gz files
    for root, _, files in os.walk(extract_to):
        for file in files:
            if file.endswith(".gz"):
                gz_path = os.path.join(root, file)
                output_path = gz_path[:-3]  # remove .gz extension
                with gzip.open(gz_path, "rb") as gz_file:
                    with open(output_path, "wb") as out_file:
                        shutil.copyfileobj(gz_file, out_file)
                os.remove(gz_path)

    os.remove(local_filename)


class TSPLibDataset(BaseDataset, TSP):
    def __init__(self):
        download_tsplib95()
        self.tsplib_dir = "tsplib95_data"
        # files = os.listdir(self.tsplib_dir)
        # self.problem_files = [file for file in files if file.endswith(".tsp")]
        self.problem_files = [
            "eil51.tsp",
            "berlin52.tsp",
            "st70.tsp",
            "eil76.tsp",
            "pr76.tsp",
            "rat99.tsp",
            "kroA100.tsp",
            "kroB100.tsp",
            "kroC100.tsp",
            "kroD100.tsp",
            "kroE100.tsp",
            "rd100.tsp",
            "eil101.tsp",
            "lin105.tsp",
            "pr124.tsp",
            "bier127.tsp",
            "ch130.tsp",
            "pr136.tsp",
            "pr144.tsp",
            "kroA150.tsp",
            "kroB150.tsp",
            "pr152.tsp",
            "u159.tsp",
            "rat195.tsp",
            "kroA200.tsp",
            "ts225.tsp",
            "tsp225.tsp",
            "pr226.tsp",
        ]

    def __getitem__(self, idx):
        problem = load(os.path.join(self.tsplib_dir, self.problem_files[idx]))
        nodes = list(problem.get_nodes())

        num_cities = problem.dimension
        if problem.node_coords or problem.display_data:
            node_coords = (
                problem.node_coords if problem.node_coords else problem.display_data
            )
            locs = []
            for _, v in node_coords.items():
                locs.append(v)
            locs = torch.tensor(locs).float()
        else:
            if problem.edge_weight_format == "FULL_MATRIX":
                dist = torch.tensor(problem.edge_weights).float()
            else:
                dist = torch.zeros((num_cities, num_cities))
                for i in range(num_cities):
                    for j in range(num_cities):
                        dist[i, j] = problem.get_weight(nodes[i], nodes[j])
            locs = dist_to_coords(dist)

        locs = normalize_coord(locs)
        dist = torch.cdist(locs, locs)
        instance = TSPInstance(problem.name, num_cities, locs, dist)

        solution = None
        path = os.path.join(self.tsplib_dir, self.problem_files[idx][:-4] + ".opt.tour")
        if os.path.exists(path):
            solution = load(path)
            tour = solution.tours[0]
            if not (problem.node_coords or problem.display_data):
                tour = [city - 1 for city in tour]
            tour = [nodes.index(i) for i in tour]
            tour = tour + [tour[0]]
            tour = torch.tensor(tour)
            solution = TSPSolution(tour=tour)

        return instance, solution

    def __len__(self):
        return len(self.problem_files)
