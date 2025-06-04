from pathlib import Path
from typing import Optional, Tuple
from src.datasets.base import BaseDataset
from src.problems.jsp.fjsp import FJSP, FJSPInstance, FJSPSolution, FJSPFileGenerator


class FJSPBenchmarksDataset(BaseDataset, FJSP):
    def __init__(self, file_pattern: Optional[str] = None):

        if file_pattern is None:
            file_pattern = "data/jsp/Brandimarte/*.fjs"

            print(Path().cwd())

        self.files_list = list(Path().glob(file_pattern))


    def __getitem__(self, idx) -> Tuple[FJSPInstance, Optional[FJSPSolution]]:

        filename = self.files_list[idx]

        generator = FJSPFileGenerator(file=filename)

        return generator.sample(), None
    
    def __len__(self):
        return len(self.files_list)