from typing import List, Optional


class BlueCast:
    def __init__(self,
                 preprocessing_steps,
                 prediction_mode: bool,
                 cat_columns: Optional[List[str]],
                 target_column: str):
        self.preprocessing_steps = preprocessing_steps
        self.prediction_mode = prediction_mode
        self.cat_columns = cat_columns
        self.target_column = target_column