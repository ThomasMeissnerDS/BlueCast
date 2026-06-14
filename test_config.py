from bluecast.ai import BlueCastAI
import pandas as pd
from unittest.mock import MagicMock

ai = BlueCastAI(api_key="123")
ai._llm = MagicMock()
df = pd.DataFrame({'class': [1, 2], 'f1': [1, 2]})

ai.run(df, "class", prompt="test", mode="ultimate", max_iterations=5)
